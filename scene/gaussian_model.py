#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import functools

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation
import torch_cluster
import pytorch3d.transforms
import sys
# from common.pose import Pose

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.pose_lr_joint = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args = None):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args

        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_from_pcd(self, pcd: BasicPointCloud, voxel_size): # , spatial_lr_scale: float
        # assert spatial_lr_scale == self.spatial_lr_scale
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Adding {fused_point_cloud.shape[0]} gaussians")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # opacities = inverse_sigmoid(1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) #### !!!!! just for debugging
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        N_old = self._xyz.shape[0]

        dist_to_existing_points = torch.cdist(fused_point_cloud, self._xyz).min(dim=1)[0]
        keep_mask = dist_to_existing_points > voxel_size 
        print(f"jk, actually only adding {keep_mask.sum()} gaussians")

        N_new = N_old + keep_mask.sum()

        def update_param(old_data: torch.Tensor, new_data: torch.Tensor):
            new_data.requires_grad_(True)
            new_data = new_data.to(old_data)[keep_mask]

            output_data = torch.zeros((N_new, *old_data.shape[1:])).to(old_data)
            output_data[:N_old] = old_data
            output_data[N_old:] = new_data
            return nn.Parameter(output_data)
        
        self._xyz = update_param(self._xyz, fused_point_cloud)
        self._features_dc = update_param(self._features_dc, features[:,:,0:1].transpose(1, 2).contiguous())
        self._features_rest = update_param(self._features_rest, features[:,:,1:].transpose(1, 2).contiguous())
        self._scaling = update_param(self._scaling, scales)
        self._rotation = update_param(self._rotation, rots)
        self._opacity = update_param(self._opacity, opacities)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # TODO: tmp Will reset accumulated gradient
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def create_from_gs(self, means, colors, covs, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = means.float().cuda()
        fused_color = RGB2SH(colors.float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        U, S, Vt = torch.svd(covs)

        ### isotropic ###
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        ### sampled ###
        # var = torch.cat((covs[:,0,0].unsqueeze(1), covs[:,1,1].unsqueeze(1), covs[:,2,2].unsqueeze(1)), 1)
        init_scale = 2. #2.
        scales = torch.log(torch.sqrt(S)*init_scale).float().cuda()

        ### isotropic ###
        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        ### sampled ###
        init_opa = 1. #1.
        opacities = inverse_sigmoid(init_opa * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        ### isotropic ###
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        ### sampled ###
        U[:,:,2] = U[:,:,2] * torch.linalg.det(U).unsqueeze(1)
        rots = pytorch3d.transforms.matrix_to_quaternion(U)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def training_setup(self, training_args, train_gaussians: bool = True, optimizable_poses = None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        def configure_param(param_list, param: nn.Parameter, lr: float, name: str):
            if lr > 0:
                if name != "poses":
                    param.requires_grad_(True)
                    param = [param]
                param_list.append({'params': param, 'lr': lr, "name": name})
            else:
                param.requires_grad_(False)
        
        param_groups = []

        if train_gaussians:
            param_groups += [
                (self._xyz, training_args.position_lr_init * self.spatial_lr_scale, "xyz"),
                (self._features_dc, training_args.feature_lr, "f_dc"),
                (self._features_rest, training_args.feature_lr / 20.0, "f_rest"),
                (self._opacity, training_args.opacity_lr, "opacity"),
                (self._scaling, training_args.scaling_lr, "scaling"),
                (self._rotation, training_args.rotation_lr, "rotation")
            ]

        if optimizable_poses is not None:
            param_groups.append((optimizable_poses, training_args.pose_lr_init, "poses"))

        l = []
        for p in param_groups:
            configure_param(l, *p)

        if len(l) == 0:
            raise RuntimeError("Can't optimize with nothing to optimize :)")

        # l = [
        #     {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
        #     {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
        #     {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        #     {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        #     {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        #     {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        # ]

        self.optimizer = torch.optim.Adam(l)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.pose_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_lr_init,
                                                    lr_final=training_args.pose_lr_final,
                                                    lr_delay_mult=training_args.pose_lr_delay_mult,
                                                    max_steps=training_args.pose_lr_max_steps)

        self.pose_lr_joint = training_args.pose_lr_joint

        # self.pose_scheduler_args = get_expon_lr_func(lr_init=0.01,
        #                                             lr_final=0.001,
        #                                             lr_delay_mult=0.01,
        #                                             max_steps=200)
        
    def update_pose_learning_rate(self, iteration, joint_optimization):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "poses":
                lr = self.pose_scheduler_args(iteration)
                if joint_optimization:
                    param_group['lr'] = self.pose_lr_joint
                else:
                    param_group['lr'] = lr
                return lr

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        print(xyz.shape)
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "poses":
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "poses":
                continue
            assert len(group["params"]) == 1
            
            if group["name"] == "poses":
                continue

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, split_clone_size, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > split_clone_size) # 0.31113

        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, split_clone_size):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= split_clone_size) # 0.31113
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune_original(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent*self.percent_dense)
        self.densify_and_split(grads, max_grad, extent*self.percent_dense)
        
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > extent*0.1 # 3.1113
            # small_points_ws = self.get_scaling.max(dim=1).values < min_scale # no min scale in original gs
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        print("prune_points: ", prune_mask.sum().item())

        torch.cuda.empty_cache()

    def densify_and_prune(self, max_grad, min_opacity, split_clone_size, max_scale, min_scale, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, split_clone_size)
        self.densify_and_split(grads, max_grad, split_clone_size)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale # 3.1113
            small_points_ws = self.get_scaling.max(dim=1).values < min_scale # 0.005 # help to reduce small artifact after disable opa reset
            prune_mask = torch.logical_or(torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws), small_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def prune_all(self):
        prune_mask = (self.get_opacity != 0).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def binary_opacity(self):
        opa = self.get_opacity

        # binary opacity
        thres=0
        opa[opa<=thres] = 0
        opa[opa>thres] = 1
        # update
        opacities_new = inverse_sigmoid(opa)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        torch.cuda.empty_cache()
    
    def reset_opacity_by_depth_image_fast(self, camera_pose, projmatrix, W, H, cx, cy, depth, thres, gamma, near_far):

        world_to_cam = camera_pose
        cam_to_world = torch.inverse(world_to_cam)
        viewmatrix = cam_to_world
        fullprojmatrix = projmatrix @ viewmatrix
        
        xyz = self.get_xyz
        opa = self.get_opacity

        # Use dist to camera center as depth
        # cam_position = world_to_cam[:3,3]
        # dist = torch.norm(xyz-cam_position, dim=1).view(-1,1) # Nx1

        xyz_hom = torch.cat((xyz, torch.ones((xyz.shape[0],1)).to("cuda")), axis=1) # Nx4
        p_hom = xyz_hom @ fullprojmatrix.T # Nx4
        p_hom = p_hom / p_hom[:,-1].view(-1,1)

        p_view = xyz_hom @ viewmatrix.T # Nx4
        mask_front = p_view[:,2] > 0 # select points in front of cam plane
        # Use z-axis as depth
        dist = p_view[:,2].view(-1,1)

        # NDC to img
        uv = p_hom[:,:2] # Nx2
        uv[:,0] = ((uv[:,0] + 1.0) * W - 1.0) * 0.5
        uv[:,1] = ((uv[:,1] + 1.0) * H - 1.0) * 0.5
        uv = torch.round(uv)

        uv[:,0]+= round(cx - (W/2.-0.5))
        uv[:,1]+= round(cy - (H/2.-0.5))

        mask_in_image = (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image

        dist_from_depth = torch.zeros_like((dist))
        uv = uv[mask_in_image].long()

        ###
        # print(mask_in_image.sum())
        # img = torch.zeros_like(depth)

        # for (uv_,dist_) in zip(uv, dist):
        #     img[0][uv_[1], uv_[0]] = torch.max(dist_, img[0][uv_[1], uv_[0]])
        # import matplotlib.pyplot as plt
        # plt.imshow(img[0].detach().cpu().numpy(), cmap='jet')
        # plt.show()
        # plt.imshow(depth[0].detach().cpu().numpy(), cmap='jet')
        # plt.show()
        ###

        dist_from_depth[mask_in_image] = depth[0][uv[:,1], uv[:,0]].view(-1,1) # Nx1
        if near_far:
            mask_thres = (torch.abs(dist_from_depth-dist) > thres).view(-1)
        else:
            mask_thres = ((dist_from_depth-dist) > thres).view(-1)
        mask = (mask_front & mask_in_image & mask_thres).view(-1,1)
        print('reset: ', mask.sum().item())
        # reset opacity
        opa[mask] = opa[mask] * gamma
        
        # update
        opacities_new = inverse_sigmoid(opa)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        torch.cuda.empty_cache()

    def loss_by_depth_image(self, camera_pose, projmatrix, W, H, cx, cy, depth, raw_pc, thres, near, far):

        world_to_cam = camera_pose
        cam_to_world = torch.inverse(world_to_cam)
        viewmatrix = cam_to_world
        fullprojmatrix = projmatrix @ viewmatrix
        
        xyz = self.get_xyz

        # Use dist to camera center as depth
        # cam_position = world_to_cam[:3,3]
        # dist = torch.norm(xyz-cam_position, dim=1).view(-1,1) # Nx1

        xyz_hom = torch.cat((xyz, torch.ones((xyz.shape[0],1)).to("cuda")), axis=1) # Nx4
        p_hom = xyz_hom @ fullprojmatrix.T # Nx4
        p_hom = p_hom / p_hom[:,-1].view(-1,1)

        p_view = xyz_hom @ viewmatrix.T # Nx4
        mask_front = p_view[:,2] > 0 # select points in front of cam plane
        # Use z-axis as depth
        dist = p_view[:,2].view(-1,1)

        # NDC to img
        uv = p_hom[:,:2] # Nx2
        uv[:,0] = ((uv[:,0] + 1.0) * W - 1.0) * 0.5
        uv[:,1] = ((uv[:,1] + 1.0) * H - 1.0) * 0.5
        uv = torch.round(uv)

        uv[:,0]+= round(cx - (W/2.-0.5))
        uv[:,1]+= round(cy - (H/2.-0.5))

        mask_in_image = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image
        mask_fov = mask_in_image & mask_front

        dist_fov = dist[mask_fov]
        dist_from_depth = torch.zeros_like((dist_fov))
        uv_in_fov = uv[mask_fov].long()
        xyz_fov = xyz[mask_fov]

        ###
        raw_pc = torch.tensor(raw_pc).cuda().float()
        xyz_hom = torch.cat((raw_pc, torch.ones((raw_pc.shape[0],1)).to("cuda")), axis=1) # Nx4
        p_hom = xyz_hom @ fullprojmatrix.T # Nx4
        p_hom = p_hom / p_hom[:,-1].view(-1,1)
        p_view = xyz_hom @ viewmatrix.T # Nx4
        dist_pc = p_view[:,2].view(-1,1)
        # NDC to img
        uv_pc = p_hom[:,:2] # Nx2
        uv_pc[:,0] = ((uv_pc[:,0] + 1.0) * W - 1.0) * 0.5
        uv_pc[:,1] = ((uv_pc[:,1] + 1.0) * H - 1.0) * 0.5
        uv_pc = torch.round(uv_pc).long()

        uv_pc[:,0]+= round(cx - (W/2.-0.5))
        uv_pc[:,1]+= round(cy - (H/2.-0.5))

        pixels_to_points = torch.full((H, W, 3), float('nan')).cuda()
        pixels_to_points[uv_pc[:, 1], uv_pc[:, 0]] = raw_pc[:]
        pixels_to_dist = torch.norm(pixels_to_points, dim=2)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(pixels_to_dist.detach().cpu().numpy(), cmap='jet')
        ax[1].imshow(depth[0].detach().cpu().numpy(), cmap='jet')
        plt.show()

        corr_xyz_fov = torch.full((xyz_fov.shape), float('nan')).cuda()
        corr_xyz_fov[:] = pixels_to_points[uv_in_fov[:,1], uv_in_fov[:,0]]

        # Thres Mask
        dist_from_depth = depth[0][uv_in_fov[:,1], uv_in_fov[:,0]].view(-1,1) # Nx1
        if near and far:
            fov_thres_mask = (torch.abs(dist_from_depth-dist_fov) > thres).view(-1)
        elif not near and far:
            fov_thres_mask = ((dist_fov-dist_from_depth) > thres).view(-1)
        elif near and not far:
            fov_thres_mask = ((dist_from_depth-dist_fov) > thres).view(-1)
        else:
            sys.exit('loss_by_depth_image should have either near or far setting to True')

        fov_nan_mask = torch.isnan(corr_xyz_fov).any(dim=1) # rm pixels wo/ depth measurement
        fov_valid_mask = ~fov_nan_mask

        valid_xyz = xyz_fov[fov_thres_mask & fov_valid_mask]
        valid_corr_xyz = corr_xyz_fov[fov_thres_mask & fov_valid_mask]

        return valid_xyz, valid_corr_xyz
    
    def mask_by_depth_image(self, camera_pose, projmatrix, W, H, depth, thres, near, far):

        world_to_cam = camera_pose
        cam_to_world = torch.inverse(world_to_cam)
        viewmatrix = cam_to_world
        fullprojmatrix = projmatrix @ viewmatrix
        
        xyz = self.get_xyz

        # Use dist to camera center as depth
        # cam_position = world_to_cam[:3,3]
        # dist = torch.norm(xyz-cam_position, dim=1).view(-1,1) # Nx1

        xyz_hom = torch.cat((xyz, torch.ones((xyz.shape[0],1)).to("cuda")), axis=1) # Nx4
        p_hom = xyz_hom @ fullprojmatrix.T # Nx4
        p_hom = p_hom / p_hom[:,-1].view(-1,1)

        p_view = xyz_hom @ viewmatrix.T # Nx4
        mask_front = p_view[:,2] > 0 # select points in front of cam plane
        # Use z-axis as depth
        dist = p_view[:,2].view(-1,1)
        print(dist.mean())
        print(dist.shape)
        

        # NDC to img
        uv = p_hom[:,:2] # Nx2
        uv[:,0] = ((uv[:,0] + 1.0) * W - 1.0) * 0.5
        uv[:,1] = ((uv[:,1] + 1.0) * H - 1.0) * 0.5
        uv = torch.round(uv)
        mask_in_image = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image

        dist_from_depth = torch.zeros_like((dist))
        uv = uv[mask_in_image].long()
        dist_from_depth[mask_in_image] = depth[0][uv[:,1], uv[:,0]].view(-1,1) # Nx1

        ###
        # print(mask_in_image.sum())
        # img = torch.zeros_like(depth)

        # for (uv_,dist_) in zip(uv, dist):
        #     img[0][uv_[1], uv_[0]] = torch.max(dist_, img[0][uv_[1], uv_[0]])
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[0].detach().cpu().numpy(), cmap='jet')
        # plt.title('reproject')
        # plt.subplot(1, 2, 2)
        # plt.imshow(depth[0].detach().cpu().numpy(), cmap='jet')
        # plt.title('depth')
        # plt.show()
        ###

        if near and far:
            mask_thres = (torch.abs(dist_from_depth-dist) > thres).view(-1)
        elif not near and far:
            mask_thres = ((dist-dist_from_depth) > thres).view(-1)
        elif near and not far:
            mask_thres = ((dist_from_depth-dist) > thres).view(-1)
        else:
            sys.exit('mask_by_depth_image should have either near or far setting to True')

        mask = (mask_front & mask_in_image & mask_thres).view(-1,1)
        return mask
        
        # print('reset: ', mask.sum().item())
        # # reset opacity
        # opa[mask] = opa[mask] * gamma
        
        # # update
        # opacities_new = inverse_sigmoid(opa)
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # self._opacity = optimizable_tensors["opacity"]

        # torch.cuda.empty_cache()
    
    def reset_opacity_by_mask(self, mask, gamma):

        opa = self.get_opacity
        
        print('reset: ', mask.sum().item())
        # reset opacity
        opa[mask] = opa[mask] * gamma
        
        # update
        opacities_new = inverse_sigmoid(opa)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        torch.cuda.empty_cache()

    def rescale_by_mask(self, mask, gamma):

        scale = self.get_scaling
        
        print('rescale: ', mask.sum().item())
        # rescale
        scale[mask] = scale[mask] * gamma
        
        # update
        scales_new = self.scaling_inverse_activation(scale)
        optimizable_tensors = self.replace_tensor_to_optimizer(scales_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

        torch.cuda.empty_cache()

    
    def reset_opacity_outside_fov(self, camera_pose, projmatrix, W, H, gamma):

        world_to_cam = camera_pose
        cam_to_world = torch.inverse(world_to_cam)
        viewmatrix = cam_to_world
        fullprojmatrix = projmatrix @ viewmatrix
        
        xyz = self.get_xyz
        opa = self.get_opacity

        xyz_hom = torch.cat((xyz, torch.ones((xyz.shape[0],1)).to("cuda")), axis=1) # Nx4
        p_hom = xyz_hom @ fullprojmatrix.T # Nx4
        p_hom = p_hom / p_hom[:,-1].view(-1,1)
        p_view = xyz_hom @ viewmatrix.T # Nx4
        mask_front = p_view[:,2] > 0 # select points in front of cam plane

        # NDC to img
        uv = p_hom[:,:2] # Nx2
        uv[:,0] = ((uv[:,0] + 1.0) * W - 1.0) * 0.5
        uv[:,1] = ((uv[:,1] + 1.0) * H - 1.0) * 0.5
        uv = torch.round(uv)
        mask_in_image = (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image
        mask_in_image *= mask_front
        mask = ~mask_in_image
        print('fov reset: ', mask.sum().item())
        # reset opacity
        opa[mask] = opa[mask] * gamma
        
        # update
        opacities_new = inverse_sigmoid(opa)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        torch.cuda.empty_cache()

    def mask_outside_fov(self, camera_pose, projmatrix, W, H):

        world_to_cam = camera_pose
        cam_to_world = torch.inverse(world_to_cam)
        viewmatrix = cam_to_world
        fullprojmatrix = projmatrix @ viewmatrix
        
        xyz = self.get_xyz

        xyz_hom = torch.cat((xyz, torch.ones((xyz.shape[0],1)).to("cuda")), axis=1) # Nx4
        p_hom = xyz_hom @ fullprojmatrix.T # Nx4
        p_hom = p_hom / p_hom[:,-1].view(-1,1)
        p_view = xyz_hom @ viewmatrix.T # Nx4
        mask_front = p_view[:,2] > 0 # select points in front of cam plane

        # NDC to img
        uv = p_hom[:,:2] # Nx2
        uv[:,0] = ((uv[:,0] + 1.0) * W - 1.0) * 0.5
        uv[:,1] = ((uv[:,1] + 1.0) * H - 1.0) * 0.5
        uv = torch.round(uv)
        mask_in_image = (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image
        mask_in_image *= mask_front
        mask = ~mask_in_image
        return mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
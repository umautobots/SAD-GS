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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import get_expon_lr_func
import pytorch3d.transforms

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 depth=None, R_gt=None, T_gt=None,
                 mat=None,
                 raw_pc=None,
                 kdtree=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.depth = depth.to(self.data_device) if depth is not None else None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if raw_pc is not None:
            self.raw_pc = raw_pc
        if kdtree is not None:
            self.kdtree = kdtree
        if mat is not None:
            self.mat = mat

        if R_gt is not None and T_gt is not None:
            _world_view_transform_gt = torch.tensor(getWorld2View2(R_gt, T_gt, trans, scale)).cuda()
            self.pose_tensor_gt = self.transform_to_tensor(_world_view_transform_gt)

        _world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        self.pose_tensor = self.transform_to_tensor(_world_view_transform)
        
        # # add noise to rot !!!!!!!!!!!!!!!!!!
        # torch.manual_seed(0)
        # self.pose_tensor[3:] += torch.rand(3).cuda() * 0.2

        self.pose_tensor.requires_grad_(True)

        ### define optimizer ###
        l = [{'params': [self.pose_tensor], 'lr': 0.005, "name": "pose"}] # 0.005
        # l = [{'params': [self.pose_tensor], 'lr': 0.001, "name": "pose"},
        #      {'params': [self.world_view_transform], 'lr': 0.001, "name": "viewmat"}
        #      {'params': [self.full_proj_transform], 'lr': 0.001, "name": "projmat"}]
        self.optimizer = torch.optim.Adam(l)

        # loc w/ color loss
        # self.pose_scheduler_args = get_expon_lr_func(lr_init=0.1,
        #                                             lr_final=0.02,
        #                                             lr_delay_mult=0.01,
        #                                             max_steps=300)
        
        self.pose_scheduler_args = get_expon_lr_func(lr_init=0.02,
                                                    lr_final=0.002,
                                                    lr_delay_mult=0.01,
                                                    max_steps=300)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose":
                lr = self.pose_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    def get_world_view_transform(self):
        world_view_transform = self.tensor_to_transform(self.pose_tensor).transpose(0,1)
        return world_view_transform
    
    def get_full_proj_transform(self):
        world_view_transform = self.tensor_to_transform(self.pose_tensor).transpose(0,1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()      
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        return full_proj_transform
    
    def get_camera_center(self):
        world_view_transform = self.tensor_to_transform(self.pose_tensor).transpose(0,1)
        return world_view_transform.inverse()[3, :3]

    ###
    def get_projection_matrix(self):
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy)

    ## Converts a 4x4 transformation matrix to the se(3) twist vector
    # Inspired by a similar NICE-SLAM function.
    # @param transformation_matrix: A pytorch 4x4 homogenous transformation matrix
    # @param device: The device for the output
    # @returns: A 6-tensor [x,y,z,r_x,r_y,r_z]
    def transform_to_tensor(self, transformation_matrix, device=None):

        gpu_id = -1
        if isinstance(transformation_matrix, np.ndarray):
            if transformation_matrix.get_device() != -1:
                if transformation_matrix.requires_grad:
                    transformation_matrix = transformation_matrix.detach()
                transformation_matrix = transformation_matrix.detach().cpu()
                gpu_id = transformation_matrix.get_device()
            elif transformation_matrix.requires_grad:
                transformation_matrix = transformation_matrix.detach()
            transformation_matrix = transformation_matrix.numpy()
        elif not isinstance(transformation_matrix, torch.Tensor):
            raise ValueError((f"Invalid argument of type {type(transformation_matrix).__name__}"
                            "passed to transform_to_tensor (Expected numpy array or pytorch tensor)"))

        R = transformation_matrix[:3, :3]
        T = transformation_matrix[:3, 3]

        rot = pytorch3d.transforms.matrix_to_axis_angle(R)

        tensor = torch.cat([T, rot]).float()
        if device is not None:
            tensor = tensor.to(device)
        elif gpu_id != -1:
            tensor = tensor.to(gpu_id)
        return tensor


    ## Converts a tensor produced by transform_to_tensor to a transformation matrix
    # Inspired by a similar NICE-SLAM function.
    # @param transformation_tensors: se(3) twist vectors
    # @returns a 4x4 homogenous transformation matrix
    def tensor_to_transform(self, transformation_tensors):

        N = len(transformation_tensors.shape)
        if N == 1:
            transformation_tensors = torch.unsqueeze(transformation_tensors, 0)
        Ts, rots = transformation_tensors[:, :3], transformation_tensors[:, 3:]
        rotation_matrices = pytorch3d.transforms.axis_angle_to_matrix(rots)
        RT = torch.cat([rotation_matrices, Ts[:, :, None]], 2)
        if N == 1:
            RT = RT[0]

        H_row = torch.zeros_like(RT[0])
        H_row[3] = 1
        RT = torch.vstack((RT, H_row))
        return RT
    

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


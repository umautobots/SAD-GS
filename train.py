############################################################
# This file is not up-to-date please use examples/train.py 
############################################################

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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, from_lowerdiag
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from scipy.spatial import KDTree # Found scipy KDTree can be slow when calling it iteratively 
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import wandb
import nvidia_smi

import open3d as o3d
import numpy as np

nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

CHUNK_SIZE = 50000
CHUNK_SIZE = 5000 # wildrgbd

def create_free_pc(cam_pos, pc, near_bound=0.5, far_bound=0.5, margin=0.05, sample_res=0.1, perturb=0, device='cpu'):
    origin = cam_pos.view(1,-1).to(device)

    dist = torch.norm(pc-origin, dim=1)
    dir = (pc-origin) / dist.view(-1,1)
        
    start_points = torch.maximum(torch.zeros_like(dist), dist-near_bound)
    end_points = dist+far_bound
    num_points = (far_bound+near_bound) / sample_res
    
    # Generate linearly spaced points without using torch.linspace
    weights = torch.arange(num_points, dtype=torch.float32) / num_points #(num_points - 1)
    weights = weights.view(1, -1).to(device)
    
    sampled_dist = start_points.view(-1, 1) + weights * (end_points - start_points).view(-1, 1)
    
    dist_perturb = perturb * (torch.rand(sampled_dist.shape).to(device) - 0.5)
    sampled_dist += dist_perturb
    
    valid_mask = (sampled_dist<(dist-margin).unsqueeze(-1)) + (sampled_dist>(dist+margin).unsqueeze(-1))
    
    # sampled_points = sampled_dist.unsqueeze(2)[:,...] @ dir.unsqueeze(1)[:,...] # slow
    sampled_points = sampled_dist.unsqueeze(2) * dir.unsqueeze(1) # fast
    sampled_points = sampled_points[valid_mask]
    
    sampled_points += origin
    return sampled_points

def query_gaussians(query_xyz, mvn, alpha, device="cpu"):
    probs = []
    for query_xyz_ in torch.split(query_xyz.to(torch.float), CHUNK_SIZE, dim=0):
        # max_prob = torch.exp(mvn.log_prob(mvn.mean)).view(1,-1) # this is inaccurate?
        # individual_probs_ = torch.exp(mvn.log_prob(query_xyz_.view(query_xyz_.shape[0], 1, -1)))
        # individual_probs_ = individual_probs_ / max_prob # normalize
        individual_probs_ = torch.exp(mvn.log_prob_unnorm(query_xyz_.view(query_xyz_.shape[0], 1, -1)))
        
        # individual_probs_ = torch.exp( (-0.5) * mahalanobis_distance(query_xyz_, xyz, covariance).T**2)
        
        # individual_probs_ = torch.where(individual_probs_ < 0.01, torch.tensor(0.0).to(device), individual_probs_)
        
        individual_probs_ = individual_probs_ * alpha.view(1,-1)

        # individual_probs_[individual_probs_<0.01] = 0
    
        # individual_probs_[individual_probs_>=0.1]=1 # this one work
        # individual_probs_[individual_probs_<0.1]=0

        probs_ = torch.sum(individual_probs_, axis=1)
        probs_ = torch.clamp(probs_, 0., 1.)
        probs.append(probs_)
        
    probs = torch.cat(probs, dim=0)

    return probs

def compute_grids(gaussian_xyz, occ_pc, free_pc, grid_size=0.2, perturb=0):

    xyz = torch.cat((gaussian_xyz, occ_pc, free_pc), dim=0)

    xyz_offset_perturb = 0.5*(torch.rand(3)).cuda() * perturb
    xyz_offset = xyz.min(dim=0)[0] - xyz_offset_perturb
    xyz_norm = xyz - xyz_offset
    grid_dim_idxs = torch.floor(xyz_norm / grid_size).long()
    n_cells_per_dim = torch.max(grid_dim_idxs, dim=0)[0] + 1

    grid_indices = grid_dim_idxs[:,2]*(n_cells_per_dim[0]*n_cells_per_dim[1]) \
                + grid_dim_idxs[:,1]*n_cells_per_dim[0] \
                + grid_dim_idxs[:,0]
                
    unique_indices, inverse_indices = grid_indices.unique(return_inverse=True)
    mapping_tensor = torch.arange(unique_indices.size(0)).to(grid_indices.device)
    grid_indices = mapping_tensor[inverse_indices]
    return grid_indices[:len(gaussian_xyz)], grid_indices[len(gaussian_xyz):len(gaussian_xyz)+len(occ_pc)], grid_indices[len(gaussian_xyz)+len(occ_pc):len(gaussian_xyz)+len(occ_pc)+len(free_pc)]

def save_command(path):
    # Join the command-line arguments to form the full command
    command = ' '.join(sys.argv)
    # Append the command to a text file
    with open(path+'/command.txt', 'a') as file:
        file.write(command + '\n')

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, \
                    pose_trans_noise=args.pose_trans_noise, single_frame_id=args.single_frame_id, \
                    voxel_size=args.voxel_size, init_w_gaussian=args.init_w_gaussian, load_ply=args.load_ply)

    ## Add noise
    # gaussians.add_noise(5000)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    total_computing_time = 0
    
    viewpoint_stack_all = scene.getTrainCameras().copy()
    
    if args.dist:
        raw_pc_map=[]
        for viewpoint_cam in viewpoint_stack_all:
            raw_pc_map.append(viewpoint_cam.raw_pc)
        raw_pc_map = np.concatenate(raw_pc_map,axis=0)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(raw_pc_map)
        # o3d_pcd = o3d_pcd.voxel_down_sample(0.05)
        raw_pc_map = np.asarray(o3d_pcd.points)
        print('raw_pc_map.shape: ', raw_pc_map.shape)
        kdtree = KDTree(raw_pc_map, leaf_size=1)

    
    for iteration in range(first_iter, opt.iterations + 1):
        tic=time.time()
        wandb_logs = {}        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_depth = viewpoint_cam.depth
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #####
        active_add=False
        if active_add and iteration>1 and iteration%100==2:
            alpha_ = torch.clone(alpha)
            
            # viewpoint_cam.Cx, viewpoint_cam.Cy
            
            o3d_depth = o3d.geometry.Image(np.array(viewpoint_cam.depth.cpu()).astype(np.float32))
            tmp = np.array(viewpoint_cam.original_image.permute(1,2,0).cpu()*255).astype(np.uint8)
            
            o3d_image = o3d.geometry.Image(np.array(viewpoint_cam.original_image.permute(1,2,0).cpu()*255).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=1., depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            o3d_pc = o3d_pc.transform(viewpoint_cam.mat)
            
            # o3d.visualization.draw_geometries([o3d_pc])
            
            means = torch.tensor(np.array(o3d_pc.points))[::50]
            colors = torch.tensor(np.array(o3d_pc.colors))[::50]
            covs = torch.tensor(np.tile(np.eye(3), (len(means), 1, 1))*0.0002) # TODO: the scale is weird
            gaussians.add_from_gs(means, colors, covs, voxel_size=0)
            
            # from scene.gaussian_model import BasicPointCloud
            # pcd = BasicPointCloud(points=means, colors=colors, normals=np.zeros((means.shape[0], 3)))
            # gaussians.add_from_pcd(pcd, voxel_size=0)
        #####

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # image, viewspace_point_tensor, visibility_filter, radii, depth, depth_var = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["depth_var"]
        image, viewspace_point_tensor, visibility_filter, radii, depth, alpha = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["alpha"]
        gaussians_count, important_score = render_pkg["gaussians_count"], render_pkg["important_score"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # set far range region to black
        # gt_image[:] = gt_image[:] * ~(gt_depth < 0) 
        # print('neg depth: ', (gt_depth < 0).sum())

        if args.TUM:
            # set pixels without depth measurement to black
            gt_image[:] = gt_image[:] * ~(gt_depth == 0) 
        
        # # ignore pixels without depth measurement
        # gt_image[:] = gt_image[:] * ~(gt_depth == 0)
        # image[:] = image[:] * ~(gt_depth == 0)

        Ll1 = l1_loss(image, gt_image)

        # plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
        # # Plotting the first image
        # plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        # plt.imshow(gt_image.permute(1,2,0).cpu().detach().numpy())
        # plt.title('gt_image')
        # # Plotting the second image
        # plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        # plt.imshow(gt_depth.cpu().detach().numpy())
        # plt.title('gt_depth')
        # plt.axis('off')
        # plt.show()
        
        loss = 0

        color_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if args.CS:
            loss += color_loss * args.CS

        gt_depth[gt_depth<0]=0 # ignore far range region
        depth = torch.clone(depth)
        depth[0][gt_depth == 0] = 0
        depth_loss = l1_loss(depth, gt_depth)
        if args.DS is not None:
            loss += depth_loss * args.DS
        if args.alpha_loss is not None:
            alpha = torch.clone(alpha)
            gt_alpha = torch.ones_like(alpha)
            gt_alpha[0][gt_depth==0]=0
            alpha[0][gt_depth!=0]=1 # only force to zero
            alpha_loss = l1_loss(gt_alpha, alpha)
            loss += alpha_loss * args.alpha_loss

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            # axs[0].imshow(gt_alpha.detach().cpu().numpy().squeeze())
            # axs[0].set_title('alpha_gt')
            # axs[1].imshow(alpha.detach().cpu().numpy().squeeze())
            # axs[1].set_title('alpha')
            # plt.tight_layout()
            # plt.show()

        margin_scale = 1. #0.5 #1.
        
        if args.cls_loss:
            cam_pos = torch.tensor(viewpoint_cam.mat[:3,3])
            raw_pc = torch.tensor(viewpoint_cam.raw_pc).to(gaussians.get_xyz.device)
            occ_pc = raw_pc
            free_pc = create_free_pc(cam_pos, raw_pc, near_bound=0.05*margin_scale, far_bound=0.05*margin_scale+0.0001, margin=0.03*margin_scale, sample_res=0.05*margin_scale, perturb=0.05*margin_scale-(0.03*margin_scale), device=gaussians.get_xyz.device)
            # free_pc = create_free_pc(cam_pos, raw_pc, near_bound=0.1, far_bound=0.1, margin=0.06, sample_res=0.1, perturb=0.04, device=gaussians.get_xyz.device)
            # near_free_pc = create_free_pc(cam_pos, raw_pc, near_bound=5., far_bound=0.0, margin=0.03, sample_res=0.2, perturb=0.2, device=gaussians.get_xyz.device)

            # free_pc = create_free_pc(cam_pos, raw_pc, near_bound=0.1, far_bound=0.101, margin=0.03, sample_res=0.025, perturb=0.025, device=gaussians.get_xyz.device)
            # near_free_pc = create_free_pc(cam_pos, raw_pc, near_bound=5., far_bound=0.0, margin=0.03, sample_res=0.2, perturb=0.2, device=gaussians.get_xyz.device)

            # print('occ_pc.shape: ', occ_pc.shape)
            # print('free_pc.shape: ', free_pc.shape)

            # # viz
            # free_pcd = o3d.geometry.PointCloud()
            # free_pcd.points = o3d.utility.Vector3dVector(free_pc.cpu().detach().numpy())
            # free_pcd.paint_uniform_color([0,0,1])
            # occ_pcd = o3d.geometry.PointCloud()
            # occ_pcd.points = o3d.utility.Vector3dVector(occ_pc.cpu().detach().numpy())
            # occ_pcd.paint_uniform_color([1,0,0])
            # input_pcd = o3d.geometry.PointCloud()
            # input_pcd.points = o3d.utility.Vector3dVector(viewpoint_cam.raw_pc)
            # input_pcd.paint_uniform_color([0,1,0])
            # gs = o3d.geometry.PointCloud()
            # gs.points = o3d.utility.Vector3dVector(gaussians.get_xyz.detach().cpu().numpy())
            # gs.paint_uniform_color([1,1,0])
            # o3d.visualization.draw_geometries([occ_pcd, free_pcd])
            
            # M = int(raw_pc.shape[0]/100.) # TODO
            M = int(raw_pc.shape[0]/100.)
            
            occ_pc = occ_pc[torch.randint(0, occ_pc.size(0), (M,))]
            free_pc = free_pc[torch.randint(0, free_pc.size(0), (M,))]

            # near_free_pc = near_free_pc[torch.randint(0, near_free_pc.size(0), (int(M/10),))]
            # free_pc = torch.vstack((free_pc, near_free_pc))
            gs_idx, occ_idx, free_idx = compute_grids(gaussians.get_xyz, occ_pc, free_pc, grid_size=args.cls_grid_size, perturb=0.) # grid_size: 0.3 for wildrgbd
            
            max_idx = torch.max(torch.max(gs_idx.max(), occ_idx.max()), free_idx.max())+1
            occ_pts_prob, free_pts_prob = torch.zeros(len(occ_pc)).cuda(), torch.zeros(len(free_pc)).cuda()
            for idx in range(max_idx):
                gs_mask = gs_idx==idx
                occ_mask = occ_idx==idx
                free_mask = free_idx==idx
                if gs_mask.sum()>0 and (occ_mask.sum()>0 or free_mask.sum()>0):
                    gaussians_xyz_ = gaussians.get_xyz[gs_mask]
                    gaussians_cov_ = from_lowerdiag(gaussians.get_covariance())[gs_mask] + torch.eye(3).view(1,3,3).cuda()*1e-5
                    gaussians_opa_ = gaussians.get_opacity[gs_mask]
                    occ_pc_ = occ_pc[occ_mask]
                    free_pc_ = free_pc[free_mask]
                    mvn_ = torch.distributions.MultivariateNormal(gaussians_xyz_, gaussians_cov_)
                    if occ_mask.sum()>0 or free_mask.sum()>0:
                        pts_prob_ = query_gaussians(torch.cat((occ_pc_, free_pc_),dim=0), mvn_, gaussians_opa_)
                        occ_pts_prob_ = pts_prob_[:len(occ_pc_)]
                        free_pts_prob_ = pts_prob_[len(occ_pc_):len(occ_pc_)+len(free_pc_)]
                        occ_pts_prob[occ_mask] = occ_pts_prob_
                        free_pts_prob[free_mask] = free_pts_prob_
                    
                    # # viz
                    # print(idx)
                    # gs = o3d.geometry.PointCloud()
                    # gs.points = o3d.utility.Vector3dVector(gaussians_xyz_.detach().cpu().numpy())
                    # gs.paint_uniform_color([1,0,0])
                    # free = o3d.geometry.PointCloud()
                    # free.points = o3d.utility.Vector3dVector(free_pc_.detach().cpu().numpy())
                    # free.paint_uniform_color([0,1,0])
                    # occ = o3d.geometry.PointCloud()
                    # occ.points = o3d.utility.Vector3dVector(occ_pc_.detach().cpu().numpy())
                    # occ.paint_uniform_color([0,0,1])
                    # o3d.visualization.draw_geometries([gs, occ])

            # mvn = torch.distributions.MultivariateNormal(gaussians.get_xyz, from_lowerdiag(gaussians.get_covariance()) + torch.eye(3).view(1,3,3).cuda()*1e-5)
            # pts_prob_ = query_gaussians(torch.cat((occ_pc, free_pc),dim=0), mvn, gaussians.get_opacity)
            # occ_pts_prob_ = pts_prob_[:len(occ_pc)]
            # free_pts_prob_ = pts_prob_[len(occ_pc):len(occ_pc)+len(free_pc)]
            
            data_balancing_weight=0 #0 #0.001
            occ_pts_prob_error = torch.abs(1 - occ_pts_prob)
            # occ_pts_prob_error = torch.relu(1 - occ_pts_prob)

            free_pts_prob_error = torch.abs(0 - free_pts_prob)
            
            # print('occ_pts_prob: ', occ_pts_prob.mean())
            # print('free_pts_prob: ', free_pts_prob.mean())
            
            # L1 LOSS
            cls_loss = occ_pts_prob_error.mean()*data_balancing_weight + free_pts_prob_error.mean()
            # cls_loss = free_pts_prob_error.mean()
            # occ_loss = occ_pts_prob_error.mean()*data_balancing_weight
            loss += cls_loss

        if args.visibility:
            volume = torch.prod(gaussians.get_scaling, dim=1)
            # print(volume.shape)
            # print(gaussians.get_opacity.squeeze().shape)
            # print(important_score.shape)

            visibale_rate = (important_score + 1e-5) / ((volume + 1e-5)  * gaussians.get_opacity.squeeze())

            # print(important_score.max().item(), important_score.min().item())
            # print(volume.max().item(), volume.min().item())
            # print(gaussians.get_opacity.squeeze().max().item(), gaussians.get_opacity.squeeze().min().item())
            print('~~~')
            print(visibale_rate.max().item(), visibale_rate.min().item())
            visibale_rate = -torch.log(visibale_rate)
            print(visibale_rate.max().item(), visibale_rate.min().item())

            visibility_loss = visibale_rate.mean()
            lambda_visibility = 0.001
            loss += visibility_loss*lambda_visibility
            print('visibility_loss: ', (visibility_loss*lambda_visibility).item())

            # sys.exit()

        if args.dist:
            # lambda_distloss = 1e1 # 1e3
            # dist_loss = 0
            # gs_idx, raw_idx, raw_idx = compute_grids(gaussians.get_xyz, raw_pc, raw_pc, grid_size=2., perturb=0.)
            # max_idx = torch.max(gs_idx.max(), raw_idx.max())
            # for idx in range(max_idx):
            #     gs_mask = gs_idx==idx
            #     raw_mask = raw_idx==idx
            #     if gs_mask.sum()>0 and raw_mask.sum()>0:
            #         gaussians_xyz_ = gaussians.get_xyz[gs_mask].float()
            #         raw_pc_ = raw_pc[raw_idx].float()
            #         distances = torch.cdist(gaussians_xyz_, raw_pc_)
            #         dist_loss += (torch.min(distances, dim=1).values**2).sum()
            # dist_loss /= raw_pc.shape[0]
            # loss += dist_loss * lambda_distloss

            # lambda_distloss = 1e1 # 1e3
            # distances = torch.cdist(gaussians.get_xyz.float(), raw_pc[::2].float())
            # dist_loss = (torch.min(distances, dim=1).values**2).mean() # somehow using the L2 loss is important here
            # loss += dist_loss * lambda_distloss
            
            lambda_distloss = 1e1
            distances, indices = kdtree.query(gaussians.get_xyz.float().detach().cpu().numpy())
            indices = indices[:,0]
            corr_pc = torch.tensor(raw_pc_map[indices]).cuda()
            thres= 0.0
            dist_loss = ((torch.relu(torch.norm(gaussians.get_xyz - corr_pc, dim=1) - thres))**2).mean() # somehow using the L2 loss is important here
            loss += dist_loss * lambda_distloss

            
        if (args.reset_opa_far or args.reset_opa_near) and iteration>1 and iteration%100==0:
            camera_pose=torch.tensor(viewpoint_cam.mat).float().cuda()
            projmatrix=viewpoint_cam.get_projection_matrix().float().cuda()
            thres = 0.05 *  margin_scale #0.1
            gamma = 0.001
            if not args.reset_opa_far and args.reset_opa_near:
                gaussians.reset_opacity_by_depth_image_fast(camera_pose, projmatrix, gt_depth.shape[1], gt_depth.shape[0], viewpoint_cam.Cx, viewpoint_cam.Cy, gt_depth.unsqueeze(0), thres, gamma, near_far=False)
            elif args.reset_opa_far and args.reset_opa_near:
                gaussians.reset_opacity_by_depth_image_fast(camera_pose, projmatrix, gt_depth.shape[1], gt_depth.shape[0], viewpoint_cam.Cx, viewpoint_cam.Cy, gt_depth.unsqueeze(0), thres, gamma, near_far=True)
            else:
                print('Error')
                sys.exit()

        if args.fov_mask and iteration>1 and iteration%100==0:
            camera_pose = torch.tensor(viewpoint_cam.mat).float().cuda()
            projmatrix = viewpoint_cam.get_projection_matrix().float().cuda()
            gamma = 0.001
            gaussians.reset_opacity_outside_fov(camera_pose, projmatrix, image.shape[2], image.shape[1], gamma)
        
        if args.full_reset_opa and iteration%100==0:
            thres = 0.05 * margin_scale #0.05 # TODO
            gamma = 0.001
            preserve_mask=None
            for view_cam_ in viewpoint_stack_all:
                camera_pose=torch.tensor(view_cam_.mat).float().cuda()
                projmatrix=view_cam_.get_projection_matrix().float().cuda()
                gt_depth_ = view_cam_.depth
                reset_depth_mask_ = gaussians.mask_by_depth_image(camera_pose, projmatrix, gt_depth_.shape[1], gt_depth_.shape[0], gt_depth_.unsqueeze(0), thres, near=True, far=True) # True
                reset_fov_mask_ = gaussians.mask_outside_fov(camera_pose, projmatrix, gt_depth_.shape[1], gt_depth_.shape[0]).view(-1,1)
                # print('---')
                # print(torch.count_nonzero(reset_depth_mask_).item() , reset_depth_mask_.squeeze())
                # print(torch.count_nonzero(reset_fov_mask_).item(), reset_fov_mask_.squeeze())
                # import sys
                # sys.exit()
                
                reset_mask_ = reset_depth_mask_ + reset_fov_mask_
                preserve_mask_ = ~reset_mask_
                if preserve_mask==None:
                    preserve_mask = preserve_mask_
                else:
                    preserve_mask += preserve_mask_
            gaussians.reset_opacity_by_mask(~preserve_mask, gamma)
                    
        if args.depth_correct and iteration%1==0:
            thres = 0.0 #0.05 # TODO
            lambda_depth_correct = 1e4 #1e1
            for view_cam_ in viewpoint_stack_all: # TODO
                camera_pose=torch.tensor(view_cam_.mat).float().cuda()
                projmatrix=view_cam_.get_projection_matrix().float().cuda()
                gt_depth_ = view_cam_.depth
                raw_pc_ = view_cam_.raw_pc
                valid_xyz, valid_corr_xyz = gaussians.loss_by_depth_image(camera_pose, projmatrix, gt_depth_.shape[1], gt_depth_.shape[0], view_cam_.Cx, view_cam_.Cy, gt_depth_.unsqueeze(0), raw_pc_, thres, near=True, far=True) # True

                # if iteration%300==0:
                #     input_pc = o3d.geometry.PointCloud()
                #     input_pc.points = o3d.utility.Vector3dVector(valid_xyz.cpu().detach().numpy())
                #     input_pc.paint_uniform_color([1,0,0])
                #     corr_pc = o3d.geometry.PointCloud()
                #     corr_pc.points = o3d.utility.Vector3dVector(valid_corr_xyz.cpu().detach().numpy())
                #     corr_pc.paint_uniform_color([0,1,0])
                #     line_set = o3d.geometry.LineSet()
                #     line_set.points = o3d.utility.Vector3dVector(np.vstack([valid_xyz.cpu().detach().numpy(), valid_corr_xyz.cpu().detach().numpy()]))
                #     correspondences = np.arange(valid_xyz.shape[0]).reshape(-1, 1)
                #     correspondences = np.hstack([correspondences, correspondences+valid_xyz.shape[0]])
                #     line_set.lines = o3d.utility.Vector2iVector(correspondences)
                #     o3d.visualization.draw_geometries([input_pc, corr_pc, line_set])
                
                depth_correct_loss = l2_loss(valid_xyz, valid_corr_xyz)

                loss += depth_correct_loss * lambda_depth_correct

        if args.scale_loss is not None:
            scale_thres = args.scale_loss # TODO
            # print(torch.max(gaussians.get_scaling, 1).values[:10])
            scale_loss = (torch.relu(gaussians.get_scaling.max(dim=1).values - scale_thres)**2).mean()
            lambda_scale = 1e2
            loss += scale_loss * lambda_scale

        if args.occlude_rescale and iteration>1 and iteration<1900 and iteration%100==0:
            avg_T = torch.zeros_like(important_score)
            unseen_mask = gaussians_count==0
            avg_T[unseen_mask] = 0
            avg_T[~unseen_mask] = important_score[~unseen_mask] / gaussians_count[~unseen_mask]
            rescale_mask = avg_T<0.05
            # gaussians.rescale_by_mask(rescale_mask, 0.5)
            gaussians.reset_opacity_by_mask(rescale_mask, 0.001)

        '''
        # depth var loss 
        lambda_depth_var = 1 #0.1
        depth_var[0][gt_depth == 0] = 0
        depth_var_loss = torch.mean(depth_var) * lambda_depth_var
        loss += depth_var_loss
        ## viz ###
        if iteration:
            f, axarr = plt.subplots(1,2)
            gt_image_np = torch.permute(gt_image, (1, 2, 0)).cpu().detach().numpy()
            render_image_np = torch.permute(image, (1, 2, 0)).cpu().detach().numpy()
            axarr[0].imshow(gt_image_np)
            axarr[1].imshow(render_image_np)
            plt.show()
        if iteration > 1000 and iteration % 10 == 0:
            f, axarr = plt.subplots(1,3)
            axarr[0].imshow(torch.squeeze(gt_depth).cpu().detach().numpy(), vmax=3, cmap='jet')
            axarr[1].imshow(torch.squeeze(depth).cpu().detach().numpy(), vmax=3, cmap='jet')
            axarr[2].imshow(torch.squeeze(depth_var).cpu().detach().numpy(), cmap='jet')
            plt.show()
            # plt.savefig('./fig/'+f'{iteration}'+'.png')
        if iteration % 10 == 0:
            gt_image_np = torch.permute(gt_image, (1, 2, 0)).cpu().detach().numpy()
            render_image_np = torch.permute(image, (1, 2, 0)).cpu().detach().numpy()
            plt.imshow(0.5*gt_image_np+0.5*render_image_np)
            # plt.show()
            if not os.path.exists('./fig/tmp/'):
                os.mkdir('./fig/tmp/')
            plt.savefig('./fig/tmp/'+f'{iteration}'+'.png')
            print('save fig')
        '''

        if args.localization or args.BA:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                if iteration % 100 == 0:
                    if args.CS:
                        print('color loss: ', "{0:.4f}".format(color_loss.item()), end =" ")
                    if args.DS:
                        print('depth loss: ', "{0:.4f}".format(depth_loss.item()), end =" ")
                    if args.cls_loss:
                        print('cls loss: ', "{0:.4f}".format(cls_loss.item()), end =" ")
                        # print('occ loss: ', "{0:.4f}".format(occ_loss.item()), end =" ")
                    if args.alpha_loss:
                        print('alpha loss: ', "{0:.4f}".format(alpha_loss.item()), end =" ")
                    if args.dist:
                        print('dist loss: ', "{0:.4f}".format(dist_loss.item()), end =" ")
                    if args.scale_loss is not None:
                        print('scale loss: ', "{0:.8f}".format(scale_loss.item()), end =" ")
                    if args.depth_correct:
                        print('depth correct loss: ', "{0:.8f}".format(depth_correct_loss.item()), end =" ")
                    print()
            if iteration == opt.iterations:
                progress_bar.close()

            toc = time.time()
            total_computing_time += toc-tic
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            tic = time.time()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_original(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) # 0.005 0.5 !!!!!!!!!!!
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity=0.005, split_clone_size=0.05, max_scale=0.1, min_scale=0.01, max_screen_size=size_threshold)

            
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                print('reset_opacity !!!')
                gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations:
                if not args.localization:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if args.BA or args.localization:
                viewpoint_cam.optimizer.step()
                viewpoint_cam.optimizer.zero_grad(set_to_none = True)
                viewpoint_cam.update_learning_rate(iteration)

            toc = time.time()
            total_computing_time += toc-tic
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if args.wandb:
            # Plot with wandb
            wandb_logs['loss'] = loss.item()
            # wandb_logs['color_loss'] = color_loss.item()
            # wandb_logs['depth_loss'] = depth_loss.item()
            # wandb_logs['depth_var_loss'] = depth_var_loss.item()
            
            if args.CS:
                wandb_logs['color_loss'] = color_loss.item()
            if args.DS:
                wandb_logs['depth_loss'] = depth_loss.item()
            if args.cls_loss:
                wandb_logs['cls_loss'] = cls_loss.item()
            if args.dist:
                wandb_logs['dist_loss'] = dist_loss.item()
            if args.scale_loss is not None:
                wandb_logs['scale_loss'] = scale_loss.item()

            wandb_logs['t'] = total_computing_time
            wandb_logs['num_gaussian'] = len(gaussians.get_xyz)
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] == "pose":
                    wandb_logs['pose_lr'] = param_group['lr']

            device_id = int(str(loss.device)[-1])
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
            wandb_logs['gpu'] = info.used / (1024**2 * 1000) # Gb

            if iteration % 10 == 0:
                wandb.log(wandb_logs, commit=False)
            wandb.log({}, commit=True)

    if args.prune_unseen:
        print('prune unseen')
        gaussians_count_sum = torch.zeros(len(gaussians.get_xyz)).cuda()
        viewpoint_stack = scene.getTrainCameras().copy()
        for viewpoint_cam in viewpoint_stack:
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
            gaussians_count, important_score = render_pkg["gaussians_count"], render_pkg["important_score"]
            gaussians_count_sum += gaussians_count
        
        unseen_mask = gaussians_count_sum < 1
        print(len(gaussians.get_xyz))
        print('print # unseen gs: ', unseen_mask.sum().item())
        gaussians.prune_points(unseen_mask)
        print(len(gaussians.get_xyz))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def save_poses(iteration, path, viewpoint_stack):
    if not os.path.exists(path+'/poses'):
        os.mkdir(path+'/poses')
    with open(path+'/poses/'+str(iteration)+'.txt', 'w') as f:
        for viewpoint_cam in viewpoint_stack:
            t_gt = viewpoint_cam.pose_tensor_gt[:3]
            t_est = viewpoint_cam.pose_tensor[:3]
            err = torch.sqrt(torch.norm(t_gt-t_est)).cpu().numpy()
            print('error: ', err)
            # f.write(str(viewpoint_cam.pose_tensor.cpu().numpy()))


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--pose_trans_noise', type=float, default=0.0)
    # parser.add_argument("--no_color_loss", action="store_true", default=False)
    # parser.add_argument("--DS", action="store_true", default=False)
    parser.add_argument("--CS", type=float, default=1.)
    parser.add_argument('--DS', type=float, default=None)
    parser.add_argument('--alpha_loss', type=float, default=None)
    parser.add_argument("--cls_loss", action="store_true", default=False)
    parser.add_argument("--cls_grid_size", type=float, default=1.)

    parser.add_argument("--reset_opa_far", action="store_true", default=False)
    parser.add_argument("--reset_opa_near", action="store_true", default=False)
    parser.add_argument("--fov_mask", action="store_true", default=False)
    parser.add_argument("--full_reset_opa", action="store_true", default=False)
    parser.add_argument("--depth_correct", action="store_true", default=False)

    parser.add_argument("--dist", action="store_true", default=False)
    parser.add_argument("--prune_unseen", action="store_true", default=False)
    # parser.add_argument("--scale_loss", action="store_true", default=False)
    parser.add_argument('--scale_loss', type=float, default=None)
    parser.add_argument("--visibility", action="store_true", default=False)
    parser.add_argument("--occlude_rescale", action="store_true", default=False)
    
    parser.add_argument("--BA", action="store_true", default=False)
    parser.add_argument("--load_ply", action="store_true", default=False)
    parser.add_argument('--init_w_gaussian', action='store_true', default=False)
    parser.add_argument('--voxel_size', type=float, default=None)
    parser.add_argument("--localization", action="store_true", default=False)
    parser.add_argument('--single_frame_id', type=list_of_ints, default=[])
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--TUM', action="store_true", default=False)
    


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    wandb.login()
    if args.wandb:
        wandb_mode='online'
    else:
        wandb_mode='disabled'
    notes = ""
    cfg=""
    name=args.model_path.split('/')[-1]
    run = wandb.init(project='gaussian_splatting', name=name, config=cfg, save_code=True, notes=notes, mode=wandb_mode)

    os.makedirs(args.model_path, exist_ok = True)
    save_command(args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    nvidia_smi.nvmlShutdown()

    # All done
    print("\nTraining complete.")

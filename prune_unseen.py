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

CHUNK_SIZE = 5000 #500000

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


def pruning(dataset, opt, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, pose_trans_noise=args.pose_trans_noise, single_frame_id=args.single_frame_id, voxel_size=args.voxel_size, init_w_gaussian=args.init_w_gaussian)

    point_cloud_path = os.path.join(args.model_path, "point_cloud/iteration_{}/point_cloud.ply".format(opt.iterations))
    gaussians.load_ply(point_cloud_path)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()

    print('prune unseen')
    gaussians_count_sum = torch.zeros(len(gaussians.get_xyz)).cuda()
    important_score_sum = torch.zeros(len(gaussians.get_xyz)).cuda()

    viewpoint_stack = scene.getTrainCameras().copy()
    for viewpoint_cam in tqdm(viewpoint_stack):
        pipe.debug = False
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        gaussians_count, important_score = render_pkg["gaussians_count"], render_pkg["important_score"]
        gaussians_count_sum += gaussians_count
        important_score_sum += important_score
        # important_score_sum = torch.max(important_score_sum, important_score)

        camera_pose=torch.tensor(viewpoint_cam.mat).float().cuda()
        projmatrix=viewpoint_cam.get_projection_matrix().float().cuda()
        out_mask = gaussians.mask_outside_fov(camera_pose, projmatrix, image.shape[2], image.shape[1])
    
    if args.hit_count:
        rm_mask = gaussians_count_sum < 1

    print(gaussians_count_sum[:10])
    print(important_score_sum[:10])

    avg_T = torch.zeros_like(gaussians_count_sum)
    unseen_mask = gaussians_count_sum==0
    avg_T[unseen_mask]=0
    avg_T[~unseen_mask] = important_score_sum[~unseen_mask] / gaussians_count_sum[~unseen_mask]

    print('avg_T: ', avg_T.max().item(), avg_T.min().item())
    # print('gaussians_count_sum: ', gaussians_count_sum.max().item(), gaussians_count_sum.min().item())
    # print('important_score_sum: ', important_score_sum.max().item(), important_score_sum.min().item())

    # max_id = torch.argmax(avg_T)
    # print('max id: ', max_id)
    # print('score: ', important_score_sum[max_id])
    # print('count: ', gaussians_count_sum[max_id])

    rm_mask = avg_T < 0.1
    # rm_mask = important_score_sum < args.thres
    # rm_mask = rm_mask + out_mask

    print('Num of gs, before: ', len(gaussians.get_xyz))
    print('Remove gs: ', rm_mask.sum().item())
    gaussians.prune_points(rm_mask)
    print('Num of gs, after: ', len(gaussians.get_xyz))

    print("\nSaving Pruned Gaussians")
    scene.save(-opt.iterations) # TODO

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
    
    parser.add_argument('--init_w_gaussian', action='store_true', default=False)
    parser.add_argument('--voxel_size', type=float, default=None)
    parser.add_argument('--single_frame_id', type=list_of_ints, default=[])

    parser.add_argument('--hit_count', action='store_true', default=False)
    parser.add_argument('--thres', type=float, default=None)


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    pruning(lp.extract(args), op.extract(args), pp.extract(args), args)

    nvidia_smi.nvmlShutdown()

    # All done
    print("\nTraining complete.")

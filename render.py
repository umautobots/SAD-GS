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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from scipy.spatial.transform import Rotation as sciR

from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2

import cv2
import matplotlib.pyplot as plt

TUM = 0

def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_depths_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depths")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")
    # depth_var_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_var")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_depths_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    # makedirs(depth_var_path, exist_ok=True)
    
    mat_path = os.path.join(model_path, name, 'mat.npy')

    mat_list=[]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        depth = results["depth"]
        alpha = results["alpha"]
        
        # depth_var = results["depth_var"]
        # depth_var[depth_var>3]==3
        # depth_var = depth_var / 3.

        gt = view.original_image[0:3, :, :]
        gt_depth = view.depth.unsqueeze(0)

        # # normalized
        # depth = depth / (depth.max() + 1e-5)
        # gt_depth = gt_depth / (gt_depth.max() + 1e-5)

        max_depth = 50.
        depth = depth / max_depth
        gt_depth = gt_depth / max_depth

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_depth, os.path.join(gt_depths_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(alpha, os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth_var, os.path.join(depth_var_path, '{0:05d}'.format(idx) + ".png"))

        # import matplotlib.pyplot as plt
        # plt.imshow(alpha.cpu().numpy().squeeze())
        # plt.tight_layout()
        # plt.show()
        
        mat_list.append(view.mat)
    mat_list = np.array(mat_list)
    print(mat_list.shape)
    np.save(mat_path, mat_list)

def render_mask_set(model_path, iteration, views, training_views, gaussians, pipeline, background, only_seen_mask=False):
    
    prefix=''
    
    seen_render_path = os.path.join(model_path, prefix, 'test_seen_masked', "ours_{}".format(iteration), "renders")
    seen_gts_path = os.path.join(model_path, prefix, 'test_seen_masked', "ours_{}".format(iteration), "gt")
    seen_normalized_depth_path = os.path.join(model_path, prefix, 'test_seen_masked', "ours_{}".format(iteration), "normalized_depth")
    seen_mask_path = os.path.join(model_path, prefix, 'test_seen_masked', "ours_{}".format(iteration), "masks")

    makedirs(seen_render_path, exist_ok=True)
    makedirs(seen_gts_path, exist_ok=True)
    makedirs(seen_normalized_depth_path, exist_ok=True)
    makedirs(seen_mask_path, exist_ok=True)
    

    render_path = os.path.join(model_path, prefix, 'test_masked', "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, prefix, 'test_masked', "ours_{}".format(iteration), "gt")
    normalized_depth_path = os.path.join(model_path, prefix, 'test_masked', "ours_{}".format(iteration), "normalized_depth")
    mask_path = os.path.join(model_path, prefix, 'test_masked', "ours_{}".format(iteration), "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normalized_depth_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    pc = None
    for idx, training_view in enumerate(tqdm(training_views)):
        # print(training_view.depth.shape)
        # print(training_view.raw_pc.shape)
        pc = torch.tensor(training_view.raw_pc).float().cuda()

        # TODO
        if TUM:
            ### mask out far pc
            camera_pose=torch.tensor(training_view.mat).float().cuda()
            projmatrix=training_view.get_projection_matrix().float().cuda()
            world_to_cam = camera_pose
            cam_to_world = torch.inverse(world_to_cam)
            viewmatrix = cam_to_world
            fullprojmatrix = projmatrix @ viewmatrix
            xyz_hom = torch.cat((pc, torch.ones((pc.shape[0],1)).to("cuda")), axis=1) # Nx4
            p_hom = xyz_hom @ fullprojmatrix.T # Nx4
            p_hom = p_hom / p_hom[:,-1].view(-1,1)
            p_view = xyz_hom @ viewmatrix.T # Nx4

            max_depth=1.5
            mask_near = p_view[:,2] < max_depth
            pc = pc[mask_near]

        # import open3d as o3d
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(pc.cpu().detach().numpy())
        # o3d.visualization.draw_geometries([o3d_pcd])
    
    mat_list = []
    for idx, view in enumerate(tqdm(views)):
        camera_pose=torch.tensor(view.mat).float().cuda()
        mat_list.append(view.mat)
        projmatrix=view.get_projection_matrix().float().cuda()
        W, H = view.depth.shape[1], view.depth.shape[0]

        world_to_cam = camera_pose
        cam_to_world = torch.inverse(world_to_cam)
        viewmatrix = cam_to_world
        fullprojmatrix = projmatrix @ viewmatrix
        
        xyz = pc
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

        uv[:,0]+= round(view.Cx - (W/2.-0.5))
        uv[:,1]+= round(view.Cy - (H/2.-0.5))

        mask_in_image = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image

        uv = uv[mask_front * mask_in_image].long()
        seen_mask = torch.zeros_like(view.depth)
        seen_mask[uv[:,1], uv[:,0]] = 1.

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(view.depth.detach().cpu().numpy(), cmap='jet')
        # ax[1].imshow(seen_mask.detach().cpu().numpy(), cmap='jet')
        # plt.show()

        # dilation
        seen_mask = seen_mask.detach().cpu().numpy()
        kernel = np.ones((5, 5), np.uint8)
        # denoise step
        seen_mask = cv2.dilate(seen_mask, kernel, iterations=3)
        seen_mask = cv2.erode(seen_mask, kernel, iterations=3)
        
        # TODO for TUM
        if TUM:
            seen_mask_eroded = cv2.erode(seen_mask, kernel, iterations=5) # For TUM        
            seen_mask_eroded[:10,:]=0
            seen_mask_eroded[-10:,:]=0
            seen_mask_eroded[:,:10]=0
            seen_mask_eroded[:,-10:]=0
            seen_mask_eroded = torch.tensor(seen_mask_eroded).unsqueeze(0).cuda().bool()
        else:
            seen_mask_eroded = torch.tensor(seen_mask).unsqueeze(0).cuda().bool()
        
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(seen_mask, cmap='jet')
        # ax[1].imshow(seen_mask_eroded, cmap='jet')
        # plt.show()
        
        seen_mask = torch.tensor(seen_mask).unsqueeze(0).cuda().bool()

        results = render(view, gaussians, pipeline, background)

        # Save results using seen mask
        rendering = results["render"] * seen_mask_eroded
        depth = results["depth"] * seen_mask_eroded
        normalized_depth = depth / (depth.max() + 1e-5)
        gt = view.original_image[0:3, :, :] * seen_mask_eroded
        torchvision.utils.save_image(rendering, os.path.join(seen_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(seen_gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normalized_depth, os.path.join(seen_normalized_depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(seen_mask_eroded*255., os.path.join(seen_mask_path, '{0:05d}'.format(idx) + ".png"))

        alpha = results["alpha"]
        predict_mask = alpha>0.5
        mask = seen_mask_eroded | predict_mask
        
        # Save results using seen + alpha mask
        rendering = results["render"] * mask
        depth = results["depth"] * mask
        normalized_depth = depth / (depth.max() + 1e-5)
        
        # TODO
        if TUM:
            gt = view.original_image[0:3, :, :] * seen_mask # Set all unseen region to black like what we did in training
        else:
            gt = view.original_image[0:3, :, :] * mask
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normalized_depth, os.path.join(normalized_depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask*255., os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_mask : bool, single_frame_id : list_of_ints, use_pseudo_cam: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print(single_frame_id)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, single_frame_id=single_frame_id, load_ply=True, use_pseudo_cam=use_pseudo_cam)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_mask:
            render_mask_set(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), scene.getTrainCameras(), gaussians, pipeline, background, only_seen_mask=True)

        if use_pseudo_cam:
             render_set(dataset.model_path, "pseudo", scene.loaded_iter, scene.getPseudoCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mask", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--single_frame_id', type=list_of_ints, default=[])
    parser.add_argument('--use_pseudo_cam', action="store_true")

    parser.add_argument("--TUM", action="store_true")    
    
    args = get_combined_args(parser)
    # import sys
    # args = parser.parse_args(sys.argv[1:])
    # print(args.single_frame_id)
    
    TUM = args.TUM

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_mask, args.single_frame_id, args.use_pseudo_cam)
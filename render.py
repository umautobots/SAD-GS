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

def render_mask_set(model_path, iteration, views, training_views, gaussians, pipeline, background, only_seen_mask=False):
    seen_render_path = os.path.join(model_path, 'test_seen_masked', "ours_{}".format(iteration), "renders")
    seen_gts_path = os.path.join(model_path, 'test_seen_masked', "ours_{}".format(iteration), "gt")
    seen_normalized_depth_path = os.path.join(model_path, 'test_seen_masked', "ours_{}".format(iteration), "normalized_depth")
    seen_mask_path = os.path.join(model_path, 'test_seen_masked', "ours_{}".format(iteration), "masks")

    makedirs(seen_render_path, exist_ok=True)
    makedirs(seen_gts_path, exist_ok=True)
    makedirs(seen_normalized_depth_path, exist_ok=True)
    makedirs(seen_mask_path, exist_ok=True)

    render_path = os.path.join(model_path, 'test_masked', "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, 'test_masked', "ours_{}".format(iteration), "gt")
    normalized_depth_path = os.path.join(model_path, 'test_masked', "ours_{}".format(iteration), "normalized_depth")
    mask_path = os.path.join(model_path, 'test_masked', "ours_{}".format(iteration), "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normalized_depth_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    pc = None
    for idx, training_view in enumerate(tqdm(training_views)):
        # print(training_view.depth.shape)
        # print(training_view.raw_pc.shape)
        pc = torch.tensor(training_view.raw_pc).float().cuda()
    
    for idx, view in enumerate(tqdm(views)):
        camera_pose=torch.tensor(view.mat).float().cuda()
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
        mask_in_image = (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < W) & (uv[:, 1] < H) # select points that can be projected to the image

        uv = uv[mask_front * mask_in_image].long()
        seen_mask = torch.zeros_like(view.depth)
        seen_mask[uv[:,1], uv[:,0]] = 1.

        # dilation
        seen_mask = seen_mask.detach().cpu().numpy()
        kernel = np.ones((5, 5), np.uint8) 
        seen_mask = cv2.dilate(seen_mask, kernel, iterations=1)
        # plt.imshow(seen_mask)
        # plt.show()
        seen_mask = torch.tensor(seen_mask).unsqueeze(0).cuda().bool()

        results = render(view, gaussians, pipeline, background)

        alpha = results["alpha"]
        predict_mask = alpha>0.5
        mask = seen_mask | predict_mask

        # Save results using seen mask
        rendering = results["render"] * seen_mask
        depth = results["depth"] * seen_mask
        normalized_depth = depth / (depth.max() + 1e-5)
        gt = view.original_image[0:3, :, :] * seen_mask
        torchvision.utils.save_image(rendering, os.path.join(seen_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(seen_gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normalized_depth, os.path.join(seen_normalized_depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(seen_mask*255., os.path.join(seen_mask_path, '{0:05d}'.format(idx) + ".png"))

        # Save results using seen + alpha mask
        rendering = results["render"] * mask
        depth = results["depth"] * mask
        normalized_depth = depth / (depth.max() + 1e-5)
        gt = view.original_image[0:3, :, :] * mask
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normalized_depth, os.path.join(normalized_depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask*255., os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_mask : bool, single_frame_id : list_of_ints):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print(single_frame_id)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, single_frame_id=single_frame_id, load_ply=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_mask:
            render_mask_set(dataset.model_path, scene.loaded_iter, scene.getTestCameras(), scene.getTrainCameras(), gaussians, pipeline, background, only_seen_mask=True)

    #     cam = scene.getTrainCameras()[0]

    #     mat = np.identity(4)
    #     r = sciR.from_euler('zyx', [[0, 0, 0]], degrees=True)
    #     R = r.as_matrix().squeeze()
    #     T = np.array([0,0,1])
    #     mat[:3,:3] = R
    #     mat[:3, 3] = T

    #     mat_origin = np.identity(4)
    #     mat_origin[:3,:3] = cam.R
    #     mat_origin[:3, 3] = cam.T

    #     mat_new = mat_origin @ mat
    #     R = mat_new[:3,:3]
    #     T = mat_new[:3,3]
        
    #     mat = [
    #     -0.9900860768919552,
    #     0.07156406496136965,
    #     -0.1208641590832548,
    #     -0.43620921902013343,
    #     -0.14046195337064574,
    #     -0.5044396907757297,
    #     0.8519454431038345,
    #     1.4291121184954942,
    #     0,
    #     0.8604761373659884,
    #     0.509490742824351,
    #     0.45124030846428437,
    #     0,
    #     0,
    #     0,
    #     1
    #   ]
    #     mat = np.linalg.inv(np.array(mat).reshape(4,4))
    #     print(mat)
    #     mat_new = mat_origin @ mat

    #     R = mat_new[:3,:3]
    #     T = mat_new[:3,3]
    #     _world_view_transform = torch.tensor(getWorld2View2(R, T)).cuda()
    #     cam.pose_tensor = cam.transform_to_tensor(_world_view_transform)

    #     render_set(dataset.model_path, "self", scene.loaded_iter, [cam], gaussians, pipeline, background)

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
    
    args = get_combined_args(parser)
    # import sys
    # args = parser.parse_args(sys.argv[1:])
    # print(args.single_frame_id)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_mask, args.single_frame_id)
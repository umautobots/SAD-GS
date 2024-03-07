import os, sys, re
import argparse
import pathlib
import pickle
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import numpy as np
import skimage
from packaging import version

import torch_cluster

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from scene import Scene, GaussianModel
from utils.general_utils import safe_state, from_lowerdiag
from utils.sh_utils import eval_sh
import yaml
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

CHUNK_SIZE = 3000

def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)

def get_grid_uniform(resolution, bound):
    """
    Get query point coordinates for marching cubes.

    Args:
        resolution (int): marching cubes resolution.

    Returns:
        (dict): points coordinates and sampled coordinates for each axis.
    """
    length = bound[:,1]-bound[:,0]
    num = (length/resolution).astype(int)
    # print("Requested Size:", num)

    x = np.linspace(bound[0][0], bound[0][1], num[0])
    y = np.linspace(bound[1][0], bound[1][1], num[1])
    z = np.linspace(bound[2][0], bound[2][1], num[2])

    xx, yy, zz = np.meshgrid(x, y, z)

    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    grid_points = torch.tensor(np.vstack(
        [xx.ravel(), yy.ravel(), zz.ravel()]).T,
        dtype=torch.float)
    return {"grid_points": grid_points, "xyz": [x, y, z]}

def is_positive_definite(matrix):
    try:
        torch.linalg.cholesky(matrix)
        return True
    except RuntimeError:
        return False

def meshing(dataset : ModelParams, iteration : int, pipeline : PipelineParams, single_frame_id : list_of_ints):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print('meshing resolution: ', args.mesh_resolution)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, single_frame_id=single_frame_id)

        # Get gaussians' position, covariance, and opacity
        xyz = gaussians._xyz
        covariances = from_lowerdiag(gaussians.get_covariance())
        opacitys = torch.sigmoid(gaussians._opacity)
        print('num of Gaussian: ', xyz.shape[0])

        # Get color
        viewpoint_stack_all = scene.getTrainCameras().copy()
        for view_cam_ in viewpoint_stack_all:
            camera_pose = torch.tensor(view_cam_.mat).float().cuda()
            camera_center = camera_pose[:3,3]
        # camera_center = torch.tensor([0,0,0]).to('cuda') # view all gaussians from center TODO: should find closest camera center for each gaussian
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
        dir_pp = (gaussians.get_xyz - camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)

        # Store gaussians for visualization
        gs_pcd = o3d.geometry.PointCloud()
        gs_pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
        gs_pcd.paint_uniform_color([0,0,1])
        if args.save_pc:
            o3d.io.write_point_cloud(f"{dataset.model_path}/map_point_cloud.pcd", gs_pcd)

        # Mask out gaussian with non positive definite covariance
        positive_definite_checks = [is_positive_definite(covariances[i]) for i in range(covariances.size(0))]
        mask = torch.tensor(positive_definite_checks)

        xyz = xyz[mask]
        covariances = covariances[mask]
        opacitys = opacitys[mask]
        colors = colors[mask]

        covariances[:] += torch.eye(3).cuda()*1e-5

        # Generate grid points
        min_corner = torch.floor(xyz.min(dim=0)[0]) - 1
        max_corner = torch.ceil(xyz.max(dim=0)[0]) + 1
        bound = torch.stack((min_corner, max_corner), dim=1)
        grid = get_grid_uniform(args.mesh_resolution, bound.cpu().numpy())
        points = grid['grid_points'].cuda()
        print("points.shape: ", points.shape)

        # Generate MultivariateNormal using all gaussians
        mvn = torch.distributions.MultivariateNormal(xyz, covariances)

        # Query GS using grid points
        print('Get Points Opacity')
        probs = []
        points_color = []
        for points_ in tqdm(torch.split(points.to(torch.float), CHUNK_SIZE, dim=0)):
            individual_probs_ = torch.exp(mvn.log_prob(points_.view(points_.shape[0], 1, -1))) * opacitys.view(1,-1)
            individual_probs_[individual_probs_<args.min_opa]=0
            probs_ = torch.sum(individual_probs_, axis=1)
            probs.append(probs_)
            if args.color_pc:
                individual_color_ = individual_probs_.unsqueeze(2) * colors.unsqueeze(0)
                color_ = torch.sum(individual_color_, axis=1) / probs_.unsqueeze(1)
                points_color.append(color_)
        probs = torch.cat(probs, dim=0)
        if args.color_pc:
            points_color = torch.cat(points_color, dim=0)

        # Mask out points having opacity below threshold
        good_pts_mask = probs > args.threshold
        good_pts = points[good_pts_mask]
        if args.color_pc:
            good_pts_colors = points_color[good_pts_mask]

        # Store final points for visualization
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(good_pts.cpu().numpy())
        if args.color_pc:
            final_pcd.colors = o3d.utility.Vector3dVector(good_pts_colors.cpu().numpy())
        if args.save_pc:
            o3d.io.write_point_cloud(f"{dataset.model_path}/sampled_point_cloud.pcd", final_pcd)
        if args.viz:
            o3d.visualization.draw_geometries([gs_pcd, final_pcd])

        # Marching cube
        print('Meshing')
        probs = probs.cpu().numpy()
        try:
            if version.parse(
                    skimage.__version__) > version.parse('0.15.0'):
                # for new version as provided in environment.yaml
                verts, faces, normals, values = skimage.measure.marching_cubes(
                    volume=probs.reshape(
                        grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                    level=args.threshold,
                    spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                grid['xyz'][1][2] - grid['xyz'][1][1],
                                grid['xyz'][2][2] - grid['xyz'][2][1]))
            else:
                # for lower version
                verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                    volume=probs.reshape(
                        grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                        grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                    level=args.threshold,
                    spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                grid['xyz'][1][2] - grid['xyz'][1][1],
                                grid['xyz'][2][2] - grid['xyz'][2][1]))
        except:
            print('marching_cubes error. Possibly no surface extracted from the level set.')

        # Convert back to world coordinates
        vertices = verts + np.array(
            [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        print('Get Color')
        vertex_colors=None
        if args.color_mesh:
            # Query GS using vertices
            vertex_colors = []
            for vertices_ in tqdm(torch.split(torch.from_numpy(vertices).to('cuda').to(torch.float), int(CHUNK_SIZE/3), dim=0)):
                individual_probs_ = torch.exp(mvn.log_prob(vertices_.view(vertices_.shape[0], 1, -1))) * opacitys.view(1,-1)
                probs_ = torch.sum(individual_probs_, axis=1)
                individual_color_ = individual_probs_.unsqueeze(2) * colors.unsqueeze(0)
                color_ = torch.sum(individual_color_, axis=1) / probs_.unsqueeze(1)
                vertex_colors.append(color_)
            vertex_colors = torch.cat(vertex_colors, dim=0).cpu().numpy()

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        if vertex_colors is not None:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        mesh_o3d.compute_vertex_normals()

        # if args.transform_to_gt_frame:
        #     mesh_o3d.transform(transformation_matrix)
        if args.viz:
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
            origin_frame.transform(camera_pose.cpu().detach().numpy())
            o3d.visualization.draw_geometries([mesh_o3d, origin_frame],
                                            mesh_show_back_face=True, mesh_show_wireframe=False)
        if args.save_mesh:
            o3d.io.write_triangle_mesh(f"{dataset.model_path}/mesh_res{args.mesh_resolution}_thres{args.threshold}_minopa{args.min_opa}.ply", mesh_o3d, compressed=False, write_vertex_colors=True, 
                                write_triangle_uvs=False, print_progress=True)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Render ground truth maps using trained nerf models")
    # parser.add_argument("configuration_path")
    # parser.add_argument("experiment_directory", type=str, help="folder in outputs with all results")
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--single_frame_id', type=list_of_ints, default=[])
    
    # params for meshing
    parser.add_argument("--mesh_resolution", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=80)
    parser.add_argument("--min_opa", type=float, default=0.1)
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--color_mesh", action="store_true",default=False, help="")
    parser.add_argument("--color_pc", action="store_true",default=False, help="")
    parser.add_argument("--save_pc", action="store_true",default=False, help="")
    parser.add_argument("--save_mesh", action="store_true",default=False, help="")
    parser.add_argument("--viz", action="store_true",default=False, help="")
    parser.add_argument("--transform_to_gt_frame", action="store_true",default=False, help="transform the mesh to gt mesh frame")
    # args = parser.parse_args()
    args = get_combined_args(parser)

    if not (args.viz or args.save_mesh or args.save_pc):
        raise RuntimeError("Either visualize or save.")

    ### Start ###
    meshing(model.extract(args), args.iteration, pipeline.extract(args), args.single_frame_id)
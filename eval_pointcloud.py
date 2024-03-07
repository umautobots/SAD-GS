from scene import Scene, GaussianModel

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import open3d as o3d
import numpy as np
import torch
import os
import argparse
import yaml

def compare_point_clouds(est_scan, gt_scan, f_score_threshold, voxel_size, save_path):
    
    print("Downsampling clouds to voxel size", voxel_size)
    est_scan = est_scan.voxel_down_sample(voxel_size)
    gt_scan = gt_scan.voxel_down_sample(voxel_size)

    print("Computing metrics")
    accuracy = np.asarray(est_scan.compute_point_cloud_distance(gt_scan))
    completion = np.asarray(gt_scan.compute_point_cloud_distance(est_scan))
    chamfer_distance = accuracy.mean() + completion.mean()

    # https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/metrics/pointcloud.py
    false_negatives = (completion > f_score_threshold).sum().item()
    false_positives = (accuracy > f_score_threshold).sum().item()
    true_positives = (len(accuracy) - false_positives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    stats = {
        "accuracy": accuracy.mean().item(),
        "completion": completion.mean().item(),
        "chamfer_distance": chamfer_distance.item(),
        "recall": recall,
        "precision": precision,
        "f-score": f_score,
        "num_points": len(accuracy)
    }
    print('-------------------')
    print(stats)
    with open(save_path, 'w+') as yaml_stats_f:
        yaml.dump(stats, yaml_stats_f, indent = 2)

def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--level", type=float, default=0.01)
    parser.add_argument("--f_score_threshold", type=float, default=0.1)
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument('--single_frame_id', type=list_of_ints, default=[])
    
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    dataset = model.extract(args)
    print(dataset.model_path)

    gt_map_path = os.path.join(dataset.model_path, 'input.ply')
    est_map_path = os.path.join(dataset.model_path, f'mesh/pc_level{args.level}.ply')
    save_path_raw = os.path.join(dataset.model_path,'mesh/eval.txt')
    save_path_rm_far = os.path.join(dataset.model_path,'mesh/eval_no_occ.txt')

    gt_map = o3d.io.read_point_cloud(gt_map_path)
    est_map = o3d.io.read_point_cloud(est_map_path)

    # o3d.visualization.draw_geometries([est_map, gt_map])

    print('single_frame_id: ', args.single_frame_id)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, pose_trans_noise=0., single_frame_id=args.single_frame_id, voxel_size=None, init_w_gaussian=False)

    means = torch.tensor(np.array(est_map.points))
    n = len(means)
    colors = torch.tensor(np.zeros((n,3)))
    covs = torch.tensor(np.zeros((n,3,3)))

    gaussians.create_from_gs(means, colors, covs, spatial_lr_scale=0.)
    print('Done loading est pointcloud')

    viewpoint_stack_all = scene.getTrainCameras().copy()
    for view_cam_ in viewpoint_stack_all:
        camera_pose=torch.tensor(view_cam_.mat).float().cuda()
        projmatrix=view_cam_.get_projection_matrix().float().cuda()
        gt_depth_ = view_cam_.depth
        thres = 0.2 # (m)
        reset_mask_ = gaussians.mask_by_depth_image(camera_pose, projmatrix, gt_depth_.shape[1], gt_depth_.shape[0], gt_depth_.unsqueeze(0), thres, near=False, far=True) # True
        reset_mask_ += gaussians.mask_outside_fov(camera_pose, projmatrix, gt_depth_.shape[1], gt_depth_.shape[0]).view(-1,1)
        preserve_mask_ = ~reset_mask_.squeeze()
        print(preserve_mask_.shape)
    
    filtered_est_pc = np.array(est_map.points)[preserve_mask_.detach().cpu().numpy()]
    print(filtered_est_pc.shape)
    filtered_est_map = o3d.geometry.PointCloud()
    filtered_est_map.points = o3d.utility.Vector3dVector(filtered_est_pc)

    # o3d.visualization.draw_geometries([est_map, gt_map])
    # o3d.visualization.draw_geometries([filtered_est_map, gt_map])

    compare_point_clouds(est_map, gt_map, args.f_score_threshold, args.voxel_size, save_path_raw)
    compare_point_clouds(filtered_est_map, gt_map, args.f_score_threshold, args.voxel_size, save_path_rm_far)

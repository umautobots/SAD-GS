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
import sys
from PIL import Image
from typing import NamedTuple
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from .gaussian_model import BasicPointCloud
import imageio
import glob
import open3d as o3d
import torch
from .initialize_utils import precompute_gaussians
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
import pandas as pd
from scipy.spatial.transform import Rotation
import json

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    Cy: float
    Cx: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    R_gt: np.array
    T_gt: np.array
    mat: np.array
    raw_pc: np.array
    kdtree: KDTree

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    pseudo_cameras: list
    nerf_normalization: dict
    ply_path: str
    gaussian_init: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    if len(cam_info)==1:
        radius = 30
        print('single frame training. Set radius to: ', radius)
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder):
    cam_infos = []
    print('images_folder: ', images_folder)
    print('depths_folder: ', depths_folder)
    
    load_depth = os.path.exists(depths_folder)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) # R_colmap.T
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
        if load_depth:
            depth_path = os.path.join(depths_folder, os.path.basename(extr.name[:-3]+'png')) # just a hack
            depth_name = os.path.basename(depth_path).split(".")[0]

        image = Image.open(image_path)
        if load_depth:
            depth = Image.open(depth_path)
        else:
            depth = np.zeros_like(image)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=depth, Cx=None, Cy=None, R_gt=None, T_gt=None, mat=None, raw_pc=None, kdtree=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    depth_dir = "depths" if depths == None else depths

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), depths_folder=os.path.join(path, depth_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    ### viz ###
    # -----
    # tf_colmap = tf inverse
    # -----
    # R_gs = R
    # T_gs = T inverse
    # -----

    viz_list=[]
    for cam_info in cam_infos[:20]:
        t = cam_info.T
        R = cam_info.R # cam_info.R = R_colmap.T (done by readColmapCameras)
        # invert t
        t = -R @ t 

        mat = np.identity(4)
        mat[:3,:3] = R
        mat[:3,3] = t
        axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis_mesh.scale(0.4, center=axis_mesh.get_center())
        mesh = axis_mesh.transform(mat)
        viz_list.append(mesh)

    axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    viz_list.append(axis_mesh)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)  
    o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    viz_list.append(o3d_pcd)
    # o3d.visualization.draw_geometries(viz_list)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           pseudo_cameras=None,
                           gaussian_init=None)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            depth_name = os.path.join(path, frame["file_path"] + "_depth_0001" + '.png')

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            print('T: ', T)

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            depth = Image.open(depth_name)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], depth=depth))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readReplicaInfo(path, eval, extension=".png", single_frame_id=None, voxel_size=None, init_w_gaussian=False, load_ply=False):
    traj_file = os.path.join(path, 'traj.txt')
    with open(traj_file, 'r') as poses_file:
        poses = poses_file.readlines()
    
    height = 680
    width = 1200
    fx = 600
    fy = 600
    cx = 599.5
    cy = 339.5

    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)
    
    image_paths = sorted(glob.glob(os.path.join(path, 'images/*')))
    depth_paths = sorted(glob.glob(os.path.join(path, 'depths/*')))
    
    cam_infos = []
    test_cam_infos = []
    mat_list=[]
    viz_list=[]
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))

    for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        mat = np.array(poses[idx].split('\n')[0].split(' ')).reshape((4,4)).astype('float64')
        mat_list.append(mat)

        R = mat[:3,:3]
        T = mat[:3, 3]

        R_gt=R.copy()
        T_gt=T.copy()

        # Invert
        T = -R.T @ T # convert from real world to GS format: R=R, T=T.inv()
        T_gt = -R_gt.T @ T_gt # convert from real world to GS format: R=R, T=T.inv()

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        depth_scaled = Image.fromarray(np.array(depth) / 6553.5 * 255.0)
        
        if len(single_frame_id)>0 and (idx not in single_frame_id):
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt,
                                mat=mat, raw_pc=None, kdtree=None)
            test_cam_infos.append(cam_info)
        else:
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=6553.5, depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            o3d_pc = o3d_pc.transform(mat)
            raw_pc = np.asarray(o3d_pc.points)
            kdtree = KDTree(raw_pc)
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt,
                                mat=mat, # Poses are w.r.t. the original frame
                                raw_pc=raw_pc,
                                kdtree=kdtree
                                )
            cam_infos.append(cam_info)
    
    train_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if not load_ply:
        for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
            if len(single_frame_id)>0 and (idx not in single_frame_id):
                continue
            mat = np.array(poses[idx].split('\n')[0].split(' ')).reshape((4,4)).astype('float64')
            image = Image.open(image_path)
            depth = Image.open(depth_path)
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            # Replica dataset cam: H: 680 W: 1200 fx: 600.0 fy: 600.0 cx: 599.5 cy: 339.5 png_depth_scale: 6553.5
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)
            # o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d_depth, intrinsic=o3d_intrinsic, extrinsic=np.identity(4), depth_scale=6553.5, stride=50)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=6553.5, depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            dist = np.linalg.norm(np.asarray(o3d_pc.points), axis=1)

            o3d_pc = o3d_pc.transform(mat)
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)), axis=0)
        
        num_pts = pc_init.shape[0]
        xyz = pc_init
        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz
        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
        print('save pcd')
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None
    
    gaussian_init = None
    if init_w_gaussian:
        mean_xyz, mean_rgb, cov = precompute_gaussians(torch.tensor(pcd.points).to('cuda'), torch.tensor(pcd.colors).to('cuda'), voxel_size)
        gaussian_init={"mean_xyz": mean_xyz, "mean_rgb": mean_rgb, "cov": cov}
    else:
        if voxel_size is not None:
            # downsample
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
            o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size)
            pc_init = np.asarray(o3d_pcd.points)
            color_init = np.asarray(o3d_pcd.colors)
            pcd = BasicPointCloud(points=pc_init, colors=color_init, normals=np.zeros((pc_init.shape[0], 3)))

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           gaussian_init=gaussian_init)
    return scene_info


def build_tum_poses_from_df(df: pd.DataFrame, zero_origin=False):
    data = torch.from_numpy(df.to_numpy(dtype=np.float64))

    ts = data[:,0]
    xyz = data[:,1:4]
    quat = data[:,4:]

    rots = torch.from_numpy(Rotation.from_quat(quat).as_matrix())
    
    poses = torch.cat((rots, xyz.unsqueeze(2)), dim=2)

    homog = torch.Tensor([0,0,0,1]).tile((poses.shape[0], 1, 1)).to(poses.device)

    poses = torch.cat((poses, homog), dim=1)

    if zero_origin:
        rot_inv = poses[0,:3,:3].T
        t_inv = -rot_inv @ poses[0,:3,3]
        start_inv = torch.hstack((rot_inv, t_inv.reshape(-1, 1)))
        start_inv = torch.vstack((start_inv, torch.tensor([0,0,0,1.0], device=start_inv.device)))
        poses = start_inv.unsqueeze(0) @ poses

    return poses.float(), ts

def readTUMInfo(path, eval, extension=".png", single_frame_id=None, voxel_size=None, init_w_gaussian=False, load_ply=False, use_pseudo_cam=False):
    folder_name = os.path.basename(path)
        
    traj_file = os.path.join(path, 'groundtruth.txt')
    ground_truth_df = pd.read_csv(traj_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
    ground_truth_df = ground_truth_df.drop(ground_truth_df[(ground_truth_df.timestamp == '#')].index)
    poses, timestamps = build_tum_poses_from_df(ground_truth_df, False)
    ts_pose = np.asarray([t for t in timestamps])

    image_ts_file = os.path.join(path, 'rgb.txt')
    depth_ts_file = os.path.join(path, 'depth.txt')
    image_data = np.loadtxt(image_ts_file, delimiter=' ', dtype=np.unicode_, skiprows=0)
    depth_data = np.loadtxt(depth_ts_file, delimiter=' ', dtype=np.unicode_, skiprows=0)
    ts_image = image_data[:, 0].astype(np.float64)
    ts_depth = depth_data[:, 0].astype(np.float64)
    
    TUM_FPS=30
    max_dt = 1.0 / TUM_FPS *  1.1

    associations = []
    ts_interval = 0.5 # (s)
    last_ts = None
    for img_idx, img_ts in enumerate(ts_image):
        depth_idx = np.argmin(np.abs(ts_depth - img_ts))
        pose_idx = np.argmin(np.abs(ts_pose - img_ts))

        if (np.abs(ts_depth[depth_idx] - img_ts) < max_dt) and \
                (np.abs(ts_pose[pose_idx] - img_ts) < max_dt):
            if last_ts !=None and (img_ts-last_ts < ts_interval):
                continue
            else:
                last_ts = img_ts
            associations.append((img_idx, depth_idx, pose_idx))
    
    cam_infos = []
    test_cam_infos = []
    pseudo_cam_infos = []

    mat_list = []
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))
    
    height = 480
    width = 640
    if 'freiburg1' in folder_name:
        print("Detect freiburg1. Use freiburg1 intrinsic.")
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    elif 'freiburg2' in folder_name:
        print("Detect freiburg2. Use freiburg2 intrinsic.")
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
    elif 'freiburg3' in folder_name:
        print("Detect freiburg3. Use freiburg3 intrinsic.")
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    depth_scale = 5000.
    
    FovY = focal2fov(fy, height) # check where to incoorporate cx, cy
    FovX = focal2fov(fx, width)

    for img_idx, depth_idx, pose_idx in associations:

        mat = np.array(poses[pose_idx])
        mat_list.append(mat)
        R = mat[:3,:3]
        T = mat[:3, 3]

        R_gt=R.copy()
        T_gt=T.copy()

        # Invert
        T = -R.T @ T # convert from real world to GS format: R=R, T=T.inv()
        T_gt = -R_gt.T @ T_gt # convert from real world to GS format: R=R, T=T.inv()
        
        image_path = f"{path}/"+image_data[img_idx][1]
        depth_path = f"{path}/"+depth_data[depth_idx][1]
        temp = Image.open(image_path)
        image = temp.copy()
        temp = Image.open(depth_path)
        depth = temp.copy()
        temp.close()
        
        ### TODO: long range filter
        max_depth = 1.5
        depth_far_mask = (np.array(depth)/depth_scale>max_depth)
        depth = np.array(depth)
        depth[depth_far_mask] = 0
        depth = Image.fromarray(depth)
        ###
        
        depth_scaled = Image.fromarray(np.array(depth) / depth_scale * 255.0)

        image_name = os.path.basename(image_path).split(".png")[0]

        if len(single_frame_id)>0 and (img_idx not in single_frame_id):
            cam_info = CameraInfo(uid=img_idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt,
                                mat=mat, raw_pc=None, kdtree=None)
            test_cam_infos.append(cam_info)
        else:
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) # w, h, fx, fy, cx, cy

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=depth_scale, depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            o3d_pc = o3d_pc.transform(mat)

            raw_pc = np.asarray(o3d_pc.points)
            kdtree = KDTree(raw_pc)
            cam_info = CameraInfo(uid=img_idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt,
                                mat=mat, # Poses are w.r.t. the original frame
                                raw_pc=raw_pc,
                                kdtree=kdtree
                                )
            cam_infos.append(cam_info)
    
    viz_list=[]

    ############################## Generate Pseudo Cam ##############################
    if use_pseudo_cam:
        relative_center = np.array([0, 0, 1])  # Center point on the Z-axis
        initial_pose = mat_list[0] #np.eye(4)  # Initial pose at the origin
        num_poses_per_side = 10  # Number of poses to generate on each side of the initial pose
        angle_change = 3  # Angle increment between consecutive poses in degrees
        radius = 1  # Radius of the circle for the camera positions
        from .gen_pseudo_cam_poses import generate_symmetric_camera_poses
        poses, center = generate_symmetric_camera_poses(initial_pose, relative_center, num_poses_per_side, angle_change, radius)

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([1, 0.706, 0])  # Gold color for visibility
        viz_list.append(center_sphere)
        
        for i, pose in enumerate(poses):
            size = 0.5 if i == 0 else 0.1
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
            mesh.transform(pose)
            viz_list.append(mesh)

            R = pose[:3,:3]
            T = pose[:3,3]
            T = -R.T @ T

            # use the first img
            image_path = f"{path}/"+image_data[0][1]
            depth_path = f"{path}/"+depth_data[0][1]
            image = Image.open(image_path)
            depth = Image.open(depth_path)

            cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=None, image_name=None, width=width, height=height, depth=depth, R_gt=R, T_gt=T,
                                mat=poses[i], raw_pc=None, kdtree=None)
            pseudo_cam_infos.append(cam_info)
    else:
        pseudo_cam_infos = None
    ##############################
    
    train_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    
    if not load_ply:
        for img_idx, depth_idx, pose_idx in associations:
            if len(single_frame_id)>0 and (img_idx not in single_frame_id):
                continue
            mat = np.array(poses[pose_idx])
            image_path = f"{path}/"+image_data[img_idx][1]
            depth_path = f"{path}/"+depth_data[depth_idx][1]
            temp = Image.open(image_path)
            image = temp.copy()
            temp = Image.open(depth_path)
            depth = temp.copy()
            temp.close()
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) # w, h, fx, fy, cx, cy
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=depth_scale, depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            dist = np.linalg.norm(np.asarray(o3d_pc.points), axis=1)

            o3d_pc = o3d_pc.transform(mat)
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)), axis=0)
                    
        num_pts = pc_init.shape[0]
        xyz = pc_init
        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz
        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
        print('save pcd')
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None
    
    gaussian_init = None
    pc_init = np.asarray(pcd.points)
    color_init = np.asarray(pcd.colors)
    
    if init_w_gaussian:
        mean_xyz, mean_rgb, cov = precompute_gaussians(torch.tensor(pcd.points).to('cuda'), torch.tensor(pcd.colors).to('cuda'), voxel_size)
        gaussian_init={"mean_xyz": mean_xyz, "mean_rgb": mean_rgb, "cov": cov}
    else:
        if voxel_size is not None:
            # downsample
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
            o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size)
            pc_init = np.asarray(o3d_pcd.points)
            color_init = np.asarray(o3d_pcd.colors)
            pcd = BasicPointCloud(points=pc_init, colors=color_init, normals=np.zeros((pc_init.shape[0], 3)))

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)  
    o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)
    viz_list.append(o3d_pcd)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=pseudo_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           gaussian_init=gaussian_init)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica" : readReplicaInfo,
    "TUM" : readTUMInfo,
}

def segment_point_cloud(voxel_grid, pcd, voxel_size):
    pcd_points = np.asarray(pcd.points)
    voxels = voxel_grid.get_voxels()
    segmented_pcds, segmented_colors = [], []
    for i in range(len(voxels)):
        index = voxels[i].grid_index
        color = voxels[i].color
        center = voxel_grid.get_voxel_center_coordinate(index)
        mask = np.all(np.abs(pcd_points - center) < voxel_size / 2, axis=1)
        segmented_pcds.append(pcd_points[mask])
        segmented_colors.append(color)
    return segmented_pcds, segmented_colors

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
from ..utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import SH2RGB
from .gaussian_model import BasicPointCloud
import imageio
import glob
import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    R_gt: np.array
    T_gt: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

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

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder):
    cam_infos = []
    print('images_folder: ', images_folder)
    print('depths_folder: ', depths_folder)
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
        R = np.transpose(qvec2rotmat(extr.qvec)) # R_colmap.T !!!!!!
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
        
        depth_path = os.path.join(depths_folder, os.path.basename(extr.name[:-3]+'png')) # just a hack
        depth_name = os.path.basename(depth_path).split(".")[0]

        image = Image.open(image_path)
        depth = Image.open(depth_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=depth)
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
                           ply_path=ply_path)
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

def readReplicaInfo(path, eval, extension=".png", pose_trans_noise=0, single_frame_id=None):
    traj_file = os.path.join(path, 'traj.txt')
    with open(traj_file, 'r') as poses_file:
        poses = poses_file.readlines()
    
    image_paths = sorted(glob.glob(os.path.join(path, 'images/*')))
    depth_paths = sorted(glob.glob(os.path.join(path, 'depths/*')))
    
    cam_infos = []
    mat_list=[]
    viz_list=[]
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))

    for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        if (single_frame_id is not None) and (idx is not single_frame_id):
            continue

        mat = np.array(poses[idx].split('\n')[0].split(' ')).reshape((4,4)).astype('float64')
        mat_list.append(mat)

        R = mat[:3,:3]
        T = mat[:3, 3]

        R_gt=R.copy()
        T_gt=T.copy()
        
        # Add noise to poses
        if pose_trans_noise > 0:
            np.random.seed(0)
            noise = np.random.rand(3) * pose_trans_noise
            T += noise

        # Invert
        T = -R.T @ T # convert from real world to GS format: R=R, T=T.inv()
        T_gt = -R_gt.T @ T_gt # convert from real world to GS format: R=R, T=T.inv()

        height = 680
        width = 1200
        focal_length_x = 600
        focal_length_y = 600
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth = Image.open(depth_path)

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=depth, R_gt=R_gt, T_gt=T_gt)
        cam_infos.append(cam_info)
    
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
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
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)[::20]), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)[::20]), axis=0)
        # downsample
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)
        o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)
        o3d_pcd = o3d_pcd.voxel_down_sample(0.2)
        pc_init = np.asarray(o3d_pcd.points)
        color_init = np.asarray(o3d_pcd.colors)

        num_pts = pc_init.shape[0]
        xyz = pc_init

        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz

        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None

    # for mat in mat_list:
    #     axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #     axis_mesh.scale(0.1, center=axis_mesh.get_center())
    #     mesh = axis_mesh.transform(mat)
    #     viz_list.append(mesh)

    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(pc_init)  
    # o3d_pcd.colors = o3d.utility.Vector3dVector(color_init)    
    # viz_list.append(o3d_pcd)

    # axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # viz_list.append(axis_mesh)

    # axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mesh = axis_mesh.translate((1,0,0))
    # axis_mesh.scale(0.5, center=axis_mesh.get_center())
    # viz_list.append(axis_mesh)

    # o3d.visualization.draw_geometries(viz_list)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica" : readReplicaInfo
}

import numpy as np
import torch
import open3d as o3d

def precompute_gaussians(xyz, rgb, grid_size=0.2):
    xyz_offset = xyz.min(dim=0)[0]
    xyz_norm = xyz - xyz_offset
    grid_dim_idxs = torch.floor(xyz_norm / grid_size).long()
    n_cells_per_dim = torch.max(grid_dim_idxs, dim=0)[0] + 1

    grid_indices = grid_dim_idxs[:,2]*(n_cells_per_dim[0]*n_cells_per_dim[1]) \
                + grid_dim_idxs[:,1]*n_cells_per_dim[0] \
                + grid_dim_idxs[:,0]
                
    unique_indices, inverse_indices = grid_indices.unique(return_inverse=True)
    mapping_tensor = torch.arange(unique_indices.size(0)).to(grid_indices.device)
    grid_indices = mapping_tensor[inverse_indices]

    n_grid_cells = grid_indices.max().item() + 1

    n_pts_per_cell = torch.zeros(n_grid_cells).long().cuda()
    mean_xyz = torch.zeros((n_grid_cells, 3)).cuda().float()
    mean_rgb = torch.zeros((n_grid_cells, 3)).cuda().float()
    cov = torch.zeros((n_grid_cells, 3, 3)).cuda().float()

    mean_xyz.index_add_(0, grid_indices, xyz.float())
    mean_rgb.index_add_(0, grid_indices, rgb.float())
    n_pts_per_cell.index_add_(0, grid_indices, torch.ones_like(grid_indices))

    mean_xyz /= n_pts_per_cell.unsqueeze(1)
    mean_rgb /= n_pts_per_cell.unsqueeze(1)

    
    centered_points = xyz - mean_xyz[grid_indices]

    cov.index_add_(0, grid_indices, (centered_points.unsqueeze(2) @ centered_points.unsqueeze(1)).float())
    cov /= n_pts_per_cell.unsqueeze(1).unsqueeze(1)-1
    cov[n_pts_per_cell == 1] = torch.eye(3).cuda()* cov[n_pts_per_cell!=1][:, [0,1,2], [0,1,2]].mean()
    

    mask = n_pts_per_cell > 0

    mean_xyz = mean_xyz[mask].detach()
    mean_rgb = mean_rgb[mask].detach()
    cov = cov[mask].detach()

    return mean_xyz, mean_rgb, cov

def naive_precompute_gaussians(xyz, rgb, grid_size=0.2):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    o3d_pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=grid_size)
    segmented_pcds, segmented_colors = segment_point_cloud(voxel_grid, o3d_pcd, voxel_size=grid_size)
    mean_list, cov_list, color_list = [], [], []

    good_segments = []
    for i in range(len(segmented_pcds)):
        mean = np.mean(segmented_pcds[i], axis=0)
        centered_data = segmented_pcds[i] - mean
        cov = np.cov(centered_data, rowvar=False)
        if cov.all() == 0:
            continue
        mean_list.append(mean)
        cov_list.append(cov)
        color_list.append(segmented_colors[i])
        good_segments.append(segmented_pcds[i])

    mean = torch.tensor(np.asarray(mean_list)).float().cuda()
    color = torch.tensor(np.asarray(color_list)).float().cuda()
    cov = torch.tensor(np.stack(cov_list, axis=0)).float().cuda()

    return mean, color, cov

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

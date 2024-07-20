import open3d as o3d
import numpy as np

def generate_symmetric_camera_poses(initial_pose, center, num_poses_per_side, angle_change, radius=1):
    """
    Generate camera poses symmetrically around a center point on the XZ-plane,
    ensuring that generation starts close to the initial pose and on the same side.
    
    Args:
    - initial_pose: np.array, the initial camera pose (4x4 matrix).
    - center: np.array, the center point (3x1 vector).
    - num_poses_per_side: int, the number of camera poses to generate on each side.
    - angle_change: float, the angle increment between consecutive poses in degrees.
    - radius: float, the radius of the circle for the camera positions.
    
    Returns:
    - List of camera poses (each is a 4x4 numpy array), including the initial pose.
    """
    poses = [initial_pose]
    total_angle_change = 0  # Initialize total angle change

    for i in range(1, num_poses_per_side + 1):
        # Increment total angle change for each pose
        total_angle_change += angle_change if angle_change != 0 else (30 * i)  # Default increment if angle_change is 0
        for direction in [-1, 0, 1]:  # -1 for left, 1 for right
            for direction2 in [-1, 0, 1]:  # -1 for down, 1 for up
                angle_rad = np.radians(total_angle_change * direction)
                angle_rad2 = np.radians(total_angle_change * direction2)
                # print('angle_rad: ', angle_rad)
                offset = np.array([np.sin(angle_rad) * radius, 0, -np.cos(angle_rad) * radius])
                
                offset = np.array([np.sin(angle_rad) * radius, np.sin(angle_rad2) * radius, -np.cos(angle_rad) * radius])
                # print(offset)
                
                position = center + offset
                direction3d = center - position
                direction3d /= np.linalg.norm(direction3d)
                
                z_axis = direction3d
                y_axis = np.array([0, 1, 0])
                x_axis = np.cross(y_axis, z_axis)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0] = x_axis
                rotation_matrix[0:3, 1] = y_axis
                rotation_matrix[0:3, 2] = z_axis
                
                pose = np.eye(4)
                pose[0:3, 0:3] = rotation_matrix[0:3, 0:3]
                pose[0:3, 3] = position
                
                poses.append(initial_pose@pose)

    new_center = initial_pose @ np.vstack((center.reshape((3,1)), np.ones((1,1))))
    new_center = new_center[:3,0]
    return poses, new_center

def visualize_camera_poses(poses, center):
    """
    Visualize camera poses and center point in Open3D.
    
    Args:
    - poses: list of np.array, the camera poses to visualize.
    - center: np.array, the center point to visualize.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Visualize the center point as a large sphere
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1, 0.706, 0])  # Gold color for visibility
    vis.add_geometry(center_sphere)
    
    for i, pose in enumerate(poses):
        size = 0.2 if i == 0 else 0.1
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        mesh.transform(pose)
        vis.add_geometry(mesh)
    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh)
    
    vis.run()
    vis.destroy_window()

# # Configuration
# center = np.array([0, 0, 1])  # Center point on the Z-axis
# initial_pose = np.eye(4)  # Initial pose at the origin
# num_poses_per_side = 5  # Number of poses to generate on each side of the initial pose
# angle_change = 10  # Angle increment between consecutive poses in degrees
# radius = 2  # Radius of the circle for the camera positions

# poses, new_center = generate_symmetric_camera_poses(initial_pose, center, num_poses_per_side, angle_change, radius)
# visualize_camera_poses(poses, center)

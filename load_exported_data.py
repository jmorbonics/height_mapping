# load_and_analyze_data.py
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def load_dataset(data_dir, mask_flag):
    """
    Load sample UR5 data.

    """
    data_dir = Path(data_dir)
    
    # Load data
    rgb_data = np.load(data_dir / 'rgb_data.npy')
    depth_data = np.load(data_dir / 'depth_data.npy')
    ee_poses = np.load(data_dir / 'ee_poses.npy')
    
    # Load camera parameters
    with open(data_dir / 'camera_intrinsics.yaml', 'r') as f:
        intrinsics = yaml.safe_load(f)
    
    with open(data_dir / 'camera_extrinsics.yaml', 'r') as f:
        extrinsics = yaml.safe_load(f)
    
    with open(data_dir / 'metadata.yaml', 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Convert metadata lists back to tuples for shape information
    metadata['rgb_shape'] = tuple(metadata['rgb_shape'])
    metadata['depth_shape'] = tuple(metadata['depth_shape'])
    metadata['ee_poses_shape'] = tuple(metadata['ee_poses_shape'])
    
    print("=== Dataset Statistics ===")
    print(f"Number of frames: {metadata['num_frames']}")
    print(f"RGB shape: {metadata['rgb_shape']}")
    print(f"Depth shape: {metadata['depth_shape']}")
    print(f"EE poses shape: {metadata['ee_poses_shape']}")
    print("\n=== Camera Parameters ===")
    print("Intrinsics:")
    for key, value in intrinsics.items():
        print(f"  {key}: {value}")
    
    # Convert lists back to numpy arrays for transformations
    R_camera_to_ee = np.array(extrinsics['R_camera_to_ee'])
    t_camera_to_ee = np.array(extrinsics['t_camera_to_ee'])

    def mask_out_ee(depth_data):
        """Mask out end-effector from depth frame."""
        mask = np.load('masking/image_mask.npy')
        for i in range(0, depth_data.shape[0]):
            depth_data[i] = depth_data[i] * mask
        return depth_data
    
    if mask_flag:
        depth_data = mask_out_ee(depth_data)
    
    # Example: Generate and transform point cloud for first frame
    points, colors = generate_point_cloud(rgb_data[0], depth_data[0], intrinsics)
    
    # Transform points from camera to end-effector frame
    points_ee = (R_camera_to_ee @ points.T + t_camera_to_ee.reshape(3, 1)).T
    
    # Transform points in first saved sample in data from end-effector to base frame using first pose
    ee_pose = ee_poses[0]
    R_ee_to_base = ee_pose[:3, :3]
    t_ee_to_base = ee_pose[:3, 3]
    points_base = (R_ee_to_base @ points_ee.T + t_ee_to_base.reshape(3, 1)).T
    
    print("\n=== Point Cloud Statistics ===")
    print(f"Number of valid points: {len(points)}")
    print(f"Point cloud bounds in base frame:")
    print(f"  X: [{points_base[:, 0].min():.3f}, {points_base[:, 0].max():.3f}]")
    print(f"  Y: [{points_base[:, 1].min():.3f}, {points_base[:, 1].max():.3f}]")
    print(f"  Z: [{points_base[:, 2].min():.3f}, {points_base[:, 2].max():.3f}]")
    print(f"Points_base: {points_base}")
    print(f"Points_base shape: {points_base.shape}")
    return {
        'rgb_data': rgb_data,
        'depth_data': depth_data,
        'ee_poses': ee_poses,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'metadata': metadata,
        'generate_point_cloud': generate_point_cloud,
        'base_points': points_base
    }


def generate_point_cloud(rgb_frame, depth_frame, intrinsics):
    """Generate point cloud from a single frame."""
    depth_scale = intrinsics['depth_scale']
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    depth = depth_frame * depth_scale
    height, width = depth.shape
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    
    # Remove invalid depth pixels
    valid = z > 0
    x = (u[valid] - cx) * z[valid] / fx
    y = (v[valid] - cy) * z[valid] / fy
    z = z[valid]
    
    points = np.vstack((x, y, z)).T
    colors = rgb_frame.reshape(-1, 3)[valid]
    
    return points, colors



if __name__ == "__main__":
    # data_dir = '/home/pranay/Repos/diffusion_policy_scooping/data/gravel_sample_data'
    data_dir = 'C:/Users/jmorb/UIUC_Coding/excavator-research/fa24/gravel_sample_data'

    dataset = load_dataset(data_dir)


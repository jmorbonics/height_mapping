from pathlib import Path
import numpy as np
import rerun as rr
import matplotlib
import sys
sys.path.append(".")
from load_exported_data import load_dataset

data_dir = 'C:/Users/jmorb/UIUC_Coding/excavator-research/fa24/output'
data_dir = Path(data_dir)
map_data = np.load(data_dir / "elevation_timesteps.npy")

cam_data_dir = 'C:/Users/jmorb/UIUC_Coding/excavator-research/fa24/gravel_sample_data'
dataset = load_dataset(cam_data_dir, mask_flag=True)


rgb = dataset["rgb_data"]

# for showing camera pose
ee_poses = dataset["ee_poses"]
intrinsics = dataset["intrinsics"]
extrinsics = dataset["extrinsics"]
metadata = dataset["metadata"]

R_camera_to_ee = np.array(extrinsics['R_camera_to_ee'])
t_camera_to_ee = np.array(extrinsics['t_camera_to_ee'])




print('\n')
print("Running Rerun Visualization Script")
print("==================================")

print("R_camera_to_ee: ", R_camera_to_ee)
print(R_camera_to_ee.shape)
print("t_camera_to_ee: ", t_camera_to_ee)
print(t_camera_to_ee.shape)


def rotation_matrix_to_vector(R):
    # Ensure R is a numpy array
    R = np.array(R)
    
    # Calculate the angle of rotation
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    # Calculate the rotation axis
    v = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))
    
    # Combine theta and v to form the 3D vector
    rotation_vector = theta * v
    return rotation_vector


print("rotation_vector: ", rotation_matrix_to_vector(R_camera_to_ee))

# Use rerun
rr.init("Elevation Mapping Visualization")
rr.spawn()
rr.set_time_seconds("stable_time", 0)
# rr.log("realsense", rr.ViewCoordinates.RDF, timeless=True)


print(map_data.shape)
for k in range(1, map_data.shape[0]):
    k_0 = k - 1
    time = k_0
    rr.set_time_seconds("stable_time", time)

    depth_image = map_data[k]
    height, width = depth_image.shape

    # Create a 3D point cloud from the depth image
    x, y = np.meshgrid(np.arange(-1.9, 0.1, 0.01), np.arange(-1, 1, 0.01))
    x = x[::-1]
    y = y[::-1]
    z = depth_image
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    colormap = matplotlib.colormaps["viridis"]
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    colors = colormap(z_normalized.flatten())[:, :3]  # Use only RGB values

    if k_0 != 0:
        rr.log("realsense/depth/point_cloud", rr.Points3D(points, colors=colors))
        rr.log("realsense/rgb/image", rr.Image(rgb[k_0*30]))
    

    ee_pose = np.reshape(ee_poses[k_0*30], (4, 4))
    # print("ee_pose: \n", ee_pose)


    R_ee_to_base = ee_pose[:3, :3]
    t_ee_to_base = ee_pose[:3, 3] + t_camera_to_ee
    color = [0, 1.0, 0.5, 0.5]

    origin = t_ee_to_base
    # rotation = rotation_matrix_to_vector(R_ee_to_base)
    rotation = R_ee_to_base @ R_camera_to_ee @ np.array([[1], [1], [1]])

    if k_0 != 0:
        rr.log("realsense/depth/arrows", rr.Arrows3D(origins=origin, vectors=rotation.T * .1, colors=color))

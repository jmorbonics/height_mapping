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
metadata = dataset["metadata"]


print('\n')
print(map_data[0])

# Use rerun
rr.init("Elevation Mapping Visualization")
rr.spawn()
rr.set_time_seconds("stable_time", 0)
rr.log("realsense", rr.ViewCoordinates.RDF, timeless=True)


pose_matrix = np.reshape(ee_poses[0], (4, 4))



print(map_data.shape)
for k in range(0, map_data.shape[0]):
    time = k * 0.03333333333333333 * 20
    rr.set_time_seconds("stable_time", time)

    depth_image = map_data[k]
    height, width = depth_image.shape

    # Create a 3D point cloud from the depth image

    x, y = np.meshgrid(np.arange(-0.1, 1.9, 0.01), np.arange(-1, 1, 0.01))
    z = depth_image
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    colormap = matplotlib.colormaps["viridis"]
    z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
    colors = colormap(z_normalized.flatten())[:, :3]  # Use only RGB values

    rr.log("realsense/depth/point_cloud", rr.Points3D(points, colors=colors))
    rr.log("realsense/rgb/image", rr.Image(rgb[k*50]))
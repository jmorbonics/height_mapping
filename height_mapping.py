import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from datetime import datetime
from scipy.stats import norm
from load_exported_data import load_dataset, generate_point_cloud
from typing import Optional, List, Dict, Tuple, Any
from grid_map import Gridmap, Length, Position

print("Environment Ready")

class Heightmap:
    def __init__(self, length: Length, resolution: float, position: Position):
        """
        Initialize heightmap which builds upon Gridmap object
        resolution: (m/cell)
        """
        self.layers = []
        self.gridmap = Gridmap()
        self.gridmap.set_geometry(length, resolution, position)
        self.gridmap.add_layer("elevation", 0)
        self.gridmap.add_layer("occupancy", 0)

    def initial_fuse(self, points):
        points = [point for point in points if point[0]**2 + point[1]**2 + point[2]**2 < 2**2]   # 3.25
        points = np.array(points)

        min_y = np.min(points[:, 1])
        self.gridmap.set_layer("elevation", min_y)
        self.fuse(points)

    def fuse(self, points):
        # Filter points
        points = [point for point in points if point[0]**2 + point[1]**2 + point[2]**2 < 2**2]   # 3.25
        points = np.array(points)

        for point in points:
            self.gridmap.update_at_position("elevation", "occupancy", Position(point[0], point[1]), point[2])
            self.gridmap.increment_at_position("occupancy", Position(point[0], point[1]))
        


if __name__ == "__main__":
    # Load data
    data_dir = 'C:/Users/jmorb/UIUC_Coding/excavator-research/fa24/gravel_sample_data'
    dataset = load_dataset(data_dir, mask_flag=True)

    # Example depth and RGB data
    depth = dataset["depth_data"]
    rgb = dataset["rgb_data"]
    ee_poses = dataset["ee_poses"]
    points_base = dataset["base_points"]
    intrinsics = dataset["intrinsics"]
    extrinsics = dataset["extrinsics"]
    metadata = dataset["metadata"]

    R_camera_to_ee = np.array(extrinsics['R_camera_to_ee'])
    t_camera_to_ee = np.array(extrinsics['t_camera_to_ee'])

    min_x = np.min(points_base[:, 0])
    max_x = np.max(points_base[:, 0])
    min_y = np.min(points_base[:, 1])
    max_y = np.max(points_base[:, 1])
    min_z = np.min(points_base[:, 2])
    max_z = np.max(points_base[:, 2])

    print("min_x: ", min_x)
    print("max_x: ", max_x)
    print("min_y: ", min_y)
    print("max_y: ", max_y)

    x_length = abs(1.5 * (max_x - min_x))
    y_length = abs(1.5 * (max_y - min_y))      # give arbitrary > 1x buffer
    length = Length(x_length, y_length)

    position = Position((max_x + min_x) / 2, (max_y + min_y) / 2)  # center of the points

    # Initialize heightmap
    heightmap = Heightmap(Length(2,2), 0.01, Position(-0.9,0))
    heightmap.initial_fuse(points_base)

    # saving layers for rerun visualization
    rerun_layers = [heightmap.gridmap.get_layer("elevation")]


    # Fuse in later frames
    for i in range(1, 1000, 50):
        points, colors = generate_point_cloud(rgb[i], depth[i], intrinsics)

        # Transform points from camera to end-effector frame
        points_ee = (R_camera_to_ee @ points.T + t_camera_to_ee.reshape(3, 1)).T
        
        # Transform points in first saved sample in data from end-effector to base frame using first pose
        ee_pose = ee_poses[i]
        R_ee_to_base = ee_pose[:3, :3]
        t_ee_to_base = ee_pose[:3, 3]
        points_base = (R_ee_to_base @ points_ee.T + t_ee_to_base.reshape(3, 1)).T

        
        heightmap.fuse(points_base)


        rerun_layers.append(copy.deepcopy(heightmap.gridmap.get_layer("elevation")))
        

    # use output/rerun_timesteps.py to visualize
    np.save("output/elevation_timesteps.npy", rerun_layers)

    heightmap.gridmap.visualize_layer("elevation")
    print("Done")


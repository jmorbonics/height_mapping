import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from load_exported_data import load_dataset
import rerun as rr
print("Environment Ready")



if __name__ == "__main__":
    # data_dir = '/home/pranay/Repos/diffusion_policy_scooping/data/gravel_sample_data'
    data_dir = 'C:/Users/jmorb/UIUC_Coding/excavator-research/fa24/gravel_sample_data'

    dataset = load_dataset(data_dir, mask_flag=True)
    print('\n')
    print(dataset["base_points"])


    # Example depth and RGB data
    depth = dataset["depth_data"]
    rgb = dataset["rgb_data"]
    ee_poses = dataset["ee_poses"]
    points_base = dataset["base_points"]
    intrinsics = dataset["intrinsics"]
    metadata = dataset["metadata"]

    # Use rerun
    rr.init("Depth and RGB Visualization")
    rr.spawn()
    rr.set_time_seconds("stable_time", 0)
    # rr.log("realsense", rr.ViewCoordinates.RDF, timeless=True)


    pose_matrix = np.reshape(ee_poses[0], (4, 4))
    rr.log(
        "realsense/",
        rr.Transform3D(
            translation=pose_matrix[:3, 3],
            mat3x3=pose_matrix[:3, :3],
            from_parent=True,
        ),
        timeless=True,
    )
    rr.log(
        "realsense/",
        rr.Pinhole(
            resolution=[metadata['rgb_shape'][2], metadata['rgb_shape'][1]],  # width, height
            focal_length=[intrinsics['fx'], intrinsics['fy']],
            principal_point=[intrinsics['cx'], intrinsics['cy']],
        ),
        timeless=True,
    )

    for k in range(0, rgb.shape[0]):
        time = k * 0.03333333333333333
        rr.set_time_seconds("stable_time", time)

        pose_matrix = np.reshape(ee_poses[k], (4, 4))
        rr.log(
            "realsense/",
            rr.Transform3D(
                translation=pose_matrix[:3, 3],
                mat3x3=pose_matrix[:3, :3],
                from_parent=True,
            ),
            timeless=True,
        )
        rr.log("realsense/rgb/image", rr.Image(rgb[k]))
        rr.log("realsense/depth/image", rr.DepthImage(depth[k], meter=1/intrinsics['depth_scale']))



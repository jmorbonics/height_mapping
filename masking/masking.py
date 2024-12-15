import numpy as np
import argparse
import cv2
from typing import Dict, Any
import sys
sys.path.append("..")
from load_exported_data import load_dataset

"""
This script loads and display RGB data from first frame of give dataset.
You then can specify a bounding box for a mask. This is used to mask out the
end effector in our 6-axis robot arm. The mask is then saved as a .png file.
"""


if __name__ == "__main__":
    data_dir = 'C:/Users/jmorb/UIUC_Coding/excavator-research/fa24/gravel_sample_data'

    dataset: Dict[str, Any] = load_dataset(data_dir, mask_flag=False)
    frame: np.ndarray = dataset["rgb_data"][0]

    print("press any key to quit")

    # cv2.imshow("Original", frame)

    # Image size is in pixels is in terminal (frames, rows, columns, rgb:3)
    # pt1: (x,y); pt2: (x,y) 
    # 0,0 corresponds to top left corner
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (250, 130), (620, 480), 255, -1)   # edit rect mask here
    # cv2.imshow("Rectangular Mask", mask)

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Mask Applied to Image", masked)

    inverted_mask = cv2.bitwise_not(mask)
    # cv2.imshow("Inverted Mask", inverted_mask)

    inverted_masked = cv2.bitwise_and(frame, frame, mask=inverted_mask)
    inverted_mask[inverted_mask == 255] = 1
    cv2.imshow("Inverted Mask Applied to Image (mask that gets saved)", inverted_masked)

    # for my purposes, inverted mask is what I want, so will save that here
    np.save("image_mask.npy", inverted_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
"""

import os
import re
from typing import List, Tuple

import numpy as np
from PIL import Image
# from tf.transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation as R

class AgricultureSVOZedLoader:
    """
    Implements some useful functions to load data from decompressed svo files.

    It loads RGB and depth images and a .csv file with pose data.

    Attributes:

    """

    def __init__(
        self,
        rgb_path: str,
        depth_path: str,
        pose_filepath: str
    ):
        """
        Initializes the decompressed SVO loader.
        
        Args:
            rgb_path: a string containing the RGB images folder path.
            depth_path: a string containing the depth images folder path.
            pose_filepath: a string containing the pose .csv file path.
        """

        self.rgb_path = rgb_path
        self.rgb_imgs = sorted(os.listdir(rgb_path))

        self.depth_path = depth_path
        self.depth_imgs = sorted(os.listdir(depth_path))

        self.depth_times = []
        for depth_file in self.depth_imgs:
            depth_time = re.search('depth_(.*)_n(.*).png', depth_file)
            depth_secs = depth_time.group(1)
            depth_nsecs = depth_time.group(2)
            self.depth_times.append(float(depth_secs) + float(depth_nsecs) / 1e9)
        self.depth_times = np.array(self.depth_times)

        self.pose_path = pose_filepath
        self.pose = np.loadtxt(
            pose_filepath,
            delimiter=',',
            dtype=float,
            skiprows=1
        )
        self.pose_times = self.pose[1:, 0] + self.pose[1:, 1] / 1e9

    def get_sync_data(self):
        """
        A generator that yields the data syncronized.
        """
        
        for rgb in self.rgb_imgs:
            rgb_time = re.search('rgb_(.*)_n(.*).png', rgb)
            rgb_secs = rgb_time.group(1)
            rgb_nsecs = rgb_time.group(2)
            rgb_time = float(rgb_secs) + float(rgb_nsecs) / 1e9
            
            rgb_pose_diff = np.array(abs(self.pose_times - rgb_time))
            rgb_depth_diff = np.array(abs(self.depth_times - rgb_time))

            best_pose = self.pose[rgb_pose_diff.argmin(), :]
            best_depth = self.depth_imgs[rgb_depth_diff.argmin()]
            
            rgb_img = np.array(Image.open(self.rgb_path + rgb).convert("RGB"))
            depth_img = np.array(Image.open(self.depth_path + best_depth))

            yield rgb_img, depth_img, best_pose

    def get_extrinsics(
        self,
        pose: List
    ) -> Tuple[List, List]:
        """
        Get the extrinsics information.

        Args:
            pose: a list containing the position and the quaternion orientation
                data in sequence (pos.x, pos.y, pos.z, orient.x, orient.y, 
                orient.z, orient.w)

        Returns:
            a list containing the robot's estimated position and orientation
            in the Euler system.
        """
        position = [pose[2], pose[3], pose[4]]
        quaternion = [pose[5], pose[6], pose[7], pose[8]]
        orientation = R.from_quat(quaternion).as_euler('xyz')

        return position, orientation
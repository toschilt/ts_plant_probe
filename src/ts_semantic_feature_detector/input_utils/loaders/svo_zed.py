"""
This module implements a class to load decompressed SVO files.
"""

import os
import re
from typing import List

import numpy as np
import numpy.typing as npt

class SVOZedLoader:
    """
    Implements some useful functions to load data from decompressed svo files.

    It loads RGB and depth images.

    Attributes:
        rgb_imgs (:obj:`list`): the sorted names of the RGB images.
        rgb_times (:obj:`np.ndarray`): the float timestamps for each
            RGB image.
        depth_imgs (:obj:`list`): the sorted names of the depth images.
        depth_times (:obj:`np.ndarray`): the float timestamps for each
            depth image.
    """

    def __init__(
        self,
        data_path: str
    ):
        """
        Initializes the decompressed SVO loader.

        Gets the images list and their timestamps.
        
        Args:
            data_path (str): the path to the the upper folder where the 
                compressed and the extracted data are storaged. It is 
                expected a folder arrangement similar to:

                /data
                    /rosbag
                        /ekf
                            - ekf.csv file
                        /imu
                            - imu.csv file
                    /svo
                        /depth
                            - PNG images
                        /rgb
                            - PNG images
        """

        svo_path = data_path + '/svo'

        rgb_path = svo_path + '/rgb'
        self.rgb_imgs = sorted(os.listdir(rgb_path))
        self.rgb_times = self._get_image_set_times(self.rgb_imgs)

        depth_path = svo_path + '/depth'
        self.depth_imgs = sorted(os.listdir(depth_path))
        self.depth_times = self._get_image_set_times(self.depth_imgs)

    def _get_image_set_times(
        self,
        image_filenames: List
    ) -> npt.ArrayLike:
        """
        Extract the timestamps from image filenames.

        Is expected that the filename has the format '($secs)_n($nsecs).png',
        where ($secs) and ($nsecs) contains the seconds and nanoseconds from
        the timestamp.

        Args:
            image_filenames (:obj:`list`): contains the images filenames.

        Returns:
            times (:obj:`np.ndarray`): the timestamp for each image storaged 
                in float format.
        """
        times = []
        for img in image_filenames:
            time = re.search('(.*)_n(.*).png', img)
            secs = time.group(1)
            nsecs = time.group(2)
            times.append(float(secs) + float(nsecs) / 1e9)
        return np.array(times)
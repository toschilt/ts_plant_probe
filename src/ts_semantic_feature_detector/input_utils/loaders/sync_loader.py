"""
"""

from typing import Dict

import numpy as np
from PIL import Image

from ts_semantic_feature_detector.input_utils.loaders.rosbags import RosbagLoader
from ts_semantic_feature_detector.input_utils.loaders.svo_zed import SVOZedLoader

class SynchronizedLoader:
    """
    Implements loading and synchronizing data from rosbags and SVO files.

    Attributes:
        data_path: a string containing the path to the the upper folder 
            where the compressed and the extracted data are storaged.
        data: a dictionary containing all the data from the extracted files.
            Images still need to be loaded.
        times: a dictionary containing all the data timestamps from the
            extracted files.  
        low_freq_topic: the topic with the least amount of data. It is used
            to synchronize the others.
    """

    def __init__(
        self,
        data_path: str
    ) -> None:
        """
        Initializes the loader.

        Args:
            data_path: a string containing the path to the the upper folder 
                where the compressed and the extracted data are storaged. It
                is expected a folder arrangement similar to:

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

        self.data_path = data_path
        rosbag_loader = RosbagLoader(data_path)
        svo_loader = SVOZedLoader(data_path)

        self.data = {
            'ekf': rosbag_loader.ekf,
            'imu': rosbag_loader.imu,
            'rgb': svo_loader.rgb_imgs,
            'depth': svo_loader.depth_imgs
        }

        self.times = {
            'ekf': rosbag_loader.ekf_times,
            'imu': rosbag_loader.imu_times,
            'rgb': svo_loader.rgb_times,
            'depth': svo_loader.depth_times
        }

        self.low_freq_topic = self._get_low_freq_topic(self.times)

    def _get_low_freq_topic(
        self,
        times: Dict
    ):
        """
        Gets the low frequency topic from all sources.

        Finds the low frequency topic by checking the size of the
        timestamp Numpy array. This topic will be used as the step
        for synchronize the others.

        Args:
            times: a dictionary containing Numpy arrays with the
                timestamps of each topic.
        """

        time_sizes = {}
        for key in times.keys():
            time_sizes[key] = times[key].shape[0]

        return min(time_sizes, key=time_sizes.get)

    def get_sync_data(
        self,
        skip_frames: int = 0
    ):
        """
        Yields the rosbag and svo data synchronized.

        Args:
            skip_frames: a integer containing the desired amount
                of frames that will be skipped at the start.
        """
        low_freq_times = self.times[self.low_freq_topic]

        index = 0
        # For each instance of the low frequency topic
        for time in low_freq_times:
            if index < skip_frames:
                index += 1
                continue

            best_fit_data = {}
            # Look for the best data that has the best time fit.
            for topic in self.times.keys():
                best_time_idx = (np.abs(self.times[topic] - time)).argmin()

                # If the topic is a image, load it.
                if topic == 'rgb' or topic == 'depth':
                    img = Image.open(
                        self.data_path + '/svo/' + topic + '/' + self.data[topic][best_time_idx]
                    )
                    best_fit_data[topic] = np.array(img)
                else:
                    best_fit_data[topic] = self.data[topic][best_time_idx]

            best_fit_data['index'] = index
            index += 1
            yield best_fit_data
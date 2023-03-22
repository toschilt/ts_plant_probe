"""
"""

import numpy as np
import numpy.typing as npt

class RosbagLoader:
    """
    Implements some useful functions to load decompressed rosbags.

    It loads EKF and IMU data.

    Attributes:
        ekf: a Numpy array containing the EKF data (timestamp, position 
            and orientation).
        ekf_times: a Numpy array containing the float timestamps for each
            EKF entry.
        imu: a Numpy array containing the IMU data (timestamp and orientation).
        imu_times: a Numpy array containing the float timestamps for each
            IMU entry.
    """

    def __init__(
        self,
        data_path: str
    ):
        """
        Initializes the rosbag loader.

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
        rosbag_path = data_path + '/rosbag'

        ekf_path = rosbag_path + '/ekf/ekf.csv'
        self.ekf, self.ekf_times = self._get_csv_and_times(ekf_path)

        imu_path = rosbag_path + '/imu/imu.csv'
        self.imu, self.imu_times = self._get_csv_and_times(imu_path)

    def _get_csv_and_times(
        self,
        csv_path: str,
    ) -> npt.ArrayLike:
        """
        Loads a CSV file and constructs the float timestamp for each entry.

        It is expected that the timestamp seconds and nanoseconds are in the
        first and second columns, respectively. It is also expected that the
        first line of the file is used for data description (it's skipped).

        Args:
            csv_path: a string containing the path to the CSV file.
        """

        data = np.loadtxt(
            csv_path,
            delimiter=',',
            dtype=float,
            skiprows=1
        )
        times = data[:, 0] + data[:, 1] / 1e9

        return data, times
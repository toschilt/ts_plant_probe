"""
"""

from typing import List, Tuple

from scipy.spatial.transform import Rotation

def get_extrinsics(
    ekf: List,
    imu: List
) -> Tuple[List, List]:
    """
    Get the extrinsics information from IMU and EKF.

    As advised by some DASLAB members, this method extracts roll
    and pitch from the IMU and yaw (heading) from EKF.

    Args:
        ekf: a list containing the EKF data. It is expected that the
            first three components are the X, Y and Z position coordinates
            and the other four components are the X, Y, Z and W orientation
            quaternion values.
        imu: a list containing the IMU data. It is expected that it has the
            X, Y, Z and W orientation quaternion values.

    Returns:
        a list containing the robot's estimated position by EFK and
    another list containing the robot's estimated orientation by EKF
    and IMU.
    """

    # Removing timestamps
    ekf = ekf[2:]
    imu = imu[2:]

    position = [ekf[0], ekf[1], ekf[2]]

    ekf_quat = [ekf[3], ekf[4], ekf[5], ekf[6]]
    imu_quat = [imu[0], imu[1], imu[2], imu[3]]
    ekf_euler = Rotation.from_quat(ekf_quat).as_euler('xyz')
    imu_euler = Rotation.from_quat(imu_quat).as_euler('xyz')
    orientation = [imu_euler[0], imu_euler[1], ekf_euler[2]]

    return position, orientation
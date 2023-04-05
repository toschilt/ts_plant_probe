"""
"""
import os

from typing import List
import numpy as np
import numpy.typing as npt

from ts_semantic_feature_detector.features_3d.scene import AgriculturalScene
from ts_semantic_feature_detector.perfomance.timer import Timer

class OutputWriter:
    """
    Compiles data from the perception algorithm to be outputted.

    Attributes:
        odom_file: a string containing the path to the odometry data.
        points_file: a string containing the path to the points data.
        separator: a string containing the separator to be used in the
            output files.
    """

    def __init__(
        self,
        odometry_file: str,
        points_file: str,
        times_file: str,
        separator: str = ',',
    ):
        """
        Initializes the compiler.

        Args:
            odometry_file: a string containing the path to the odometry data.
            points_file: a string containing the path to the points data.
            separator: a string containing the separator to be used in the
                output files.
        """

        self.odom_file = odometry_file
        self.points_file = points_file
        self.times_file = times_file
        self.separator = separator

        # Create files or clear them if they already exist
        with open(self.odom_file, 'w+') as f:
            f.write('')
        with open(self.points_file, 'w+') as f:
            f.write('')
        with open(self.times_file, 'w+') as f:
            f.write('')

    def write_odometry_factors(
        self,
        scene_id: int,
        ekf_data: List
    ):
        """
        Writes the robot pose to the odometry file.

        Args:
            scene_id: an integer containing the scene id.
            ekf_data: a list containing the robot pose.
        """

        with open(self.odom_file, 'a') as f:
            f.write(
                f'{str(scene_id)}{self.separator}'
                f'{str(ekf_data[2])}{self.separator}'
                f'{str(ekf_data[3])}{self.separator}'
                f'{str(ekf_data[4])}{self.separator}'
                f'{str(ekf_data[5])}{self.separator}'
                f'{str(ekf_data[6])}{self.separator}'
                f'{str(ekf_data[7])}{self.separator}'
                f'{str(ekf_data[8])}\n'
            )

    def write_emerging_points(
        self,
        scene_id: int,
        scene: AgriculturalScene,
    ):
        """
        Writes the emerging point to the points file.

        Args:
            scene_id: an integer containing the scene id.
            points: a numpy array containing the points.
        """
        for crop in scene.crop_group.crops:
            cluster = crop.cluster

            if cluster != -1:
                point = crop.emerging_point_local_frame

                with open(self.points_file, 'a') as f:
                    f.write(
                        f'{str(scene_id)}{self.separator}'
                        f'{str(cluster)}{self.separator}'
                        f'{str(point[0])}{self.separator}'
                        f'{str(point[1])}{self.separator}'
                        f'{str(point[2])}\n'
                    )

    def write_times(
        self,
        timer: Timer
    ):
        """
        Writes the times to the times file.
        
        Args:
            timer: a Timer object containing the times.
        """

        if os.path.getsize(self.times_file) == 0:
            with open(self.times_file, 'a') as f:
                for key in timer.measurements.keys():
                    f.write(f'{key}{self.separator}')
            
            with open(self.times_file, 'a') as f:
                f.write('\n')

        with open(self.times_file, 'a') as f:
            for key in timer.measurements.keys():
                f.write(f'{str(timer.measurements[key][-1])}{self.separator}')
            f.write('\n')
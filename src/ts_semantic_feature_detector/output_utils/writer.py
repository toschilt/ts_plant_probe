"""
Implements writing functionality to output data.
"""

import os

from typing import List
import numpy as np
import numpy.typing as npt

from ts_semantic_feature_detector.features_3d.sequence import AgriculturalSequence
from ts_semantic_feature_detector.perfomance.timer import Timer

class OutputWriter:
    """
    Compiles data from the perception algorithm to be outputted.

    Attributes:
        odom_file (str): the path to the odometry data.
        points_file (str): the path to the points data.
        separator (str): the separator to be used in the output files.
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
            odometry_file (str): the path to the odometry data.
            points_file (str): the path to the points data.
            separator (str, optional): the separator to be used in the 
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
    ) -> None:
        """
        Writes the robot pose to the odometry file.

        Args:
            scene_id (int): the scene id.
            ekf_data (:obj:`list`): the robot pose.
        """

        with open(self.odom_file, 'a') as f:
            f.write(
                f'{str(scene_id)}{self.separator}'
                # Position
                f'{str(ekf_data[2])}{self.separator}'
                f'{str(ekf_data[3])}{self.separator}'
                f'{str(ekf_data[4])}{self.separator}'
                # Orientation
                f'{str(ekf_data[5])}{self.separator}'
                f'{str(ekf_data[6])}{self.separator}'
                f'{str(ekf_data[7])}{self.separator}'
                f'{str(ekf_data[8])}{self.separator}'
                # Pose covariance
                f'{str(ekf_data[9])}{self.separator}'
                f'{str(ekf_data[10])}{self.separator}'
                f'{str(ekf_data[11])}{self.separator}'
                f'{str(ekf_data[12])}{self.separator}'
                f'{str(ekf_data[13])}{self.separator}'
                f'{str(ekf_data[14])}\n'
            )

    def write_emerging_points(
        self,
        sequence: AgriculturalSequence,
        scenes_idxs: List[int],
    ) -> None:
        """
        Writes the emerging point to the points file.

        Args:
            sequence (:obj:`features_3d.sequence.AgriculturalSequence`): the
                sequence object.
            scenes_idxs (:obj:`list`): the list of scene indexes to be written.
        """
        for id in scenes_idxs:
            scene = sequence.scenes[id]
            for crop in scene.crop_group.crops:
                # Only crops that are not outliers are written.
                if crop.cluster != -1:
                    point = crop.emerging_point_local_frame
        
                    with open(self.points_file, 'a') as f:
                        f.write(
                            f'{str(scene.index)}{self.separator}'
                            f'{str(crop.cluster.id)}{self.separator}'
                            f'{str(point[0])}{self.separator}'
                            f'{str(point[1])}{self.separator}'
                            f'{str(point[2])}\n'
                        )

    def write_times(
        self,
        timer: Timer
    ) -> None:
        """
        Writes the times to the times file.
        
        Args:
            timer (:obj:`perfomance.timer.Timer`): the timer object with the
                time measurements.
        """

        if os.path.getsize(self.times_file) == 0:
            with open(self.times_file, 'a') as f:
                for key in timer.measurements.keys():
                    f.write(f'{key}{self.separator}')
                f.write('total')
                f.write('\n')

        with open(self.times_file, 'a') as f:
            total = 0
            for key in timer.measurements.keys():
                f.write(f'{str(timer.measurements[key][-1])}{self.separator}')
                total += timer.measurements[key][-1]
            f.write(f'{str(total)}')
            f.write('\n')

    def finish(
        self,
        sequence: AgriculturalSequence,
    ):
        """
        Writes all remaining scenes to the emerging points file.

        Args:
            sequence (:obj:`features_3d.sequence.AgriculturalSequence`): the
                sequence object.
        """

        scene_idxs = [i for i in range(len(sequence.scenes))]
        self.write_emerging_points(sequence, scene_idxs)
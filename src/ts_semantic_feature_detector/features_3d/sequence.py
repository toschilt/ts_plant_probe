"""
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN

from ts_semantic_feature_detector.features_3d.camera import StereoCamera
from ts_semantic_feature_detector.features_3d.scene import AgriculturalScene

class AgriculturalSequence:
    """
    Abstracts a agriculture sequence.

    Attributes:
        scenes: a list of features_3d.scene.AgriculturalScene objects
            containing all the information from each scene in the
            sequence.
    """

    def __init__(
        self,
        scenes: List = None
    ):
        """
        Initializes the agricultural sequence.

        Args:
            scenes: a list of features_3d.scene.AgriculturalScene objects
            containing all the information from each scene in the
            sequence. If it is None, a empty list is initialzed.
        """
    
        self.scenes = []
        if scenes is not None:
            self.scenes = scenes

    def add_scene(
        self,
        scene: AgriculturalScene,
    ):
        """
        Adds a scene to this sequence.

        Args:
            scene: a features_3d.scene.AgriculturalScene object to be
                added.
        """
        for old_scene in self.scenes:
            old_scene.age += 1

        self.scenes.append(scene)

    def cluster_crops(
        self,
        eps: float = 0.05,
        min_samples: int = 3
    ) -> List:
        """
        Fits a unsupervised model to crop data to try to approximate stems.

        #TODO: Save computational power by saving the last seen scene index.

        Returns:
            a list containing the labels of the analysed crops.
        """

        descriptors = []
        for scene in self.scenes:
            for crop in scene.crop_group.crops:
                emerging_point = crop.emerging_point
                # angles = crop.crop_vector_angles

                descriptor = emerging_point[:2]
                descriptors.append(descriptor)

        descriptors = np.array(descriptors)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(descriptors)
        clusters = list(dbscan.labels_)

        i = 0
        for scene in self.scenes:
            for crop in scene.crop_group.crops:
                crop.cluster = clusters[i]
                i += 1

        return clusters
    
    def remove_old_scenes(
        self,
        max_age: int = 200
    ):
        """
        Removes old scenes from the sequence.

        Args:
            max_age: the maximum age of a scene to be removed.
        """
        self.scenes = [scene for scene in self.scenes if scene.age < max_age]

    def plot(
        self,
        data_plot: List = None,
        line_scalars: npt.ArrayLike = None,
        plane_scalars: Tuple[npt.ArrayLike, npt.ArrayLike] = None,
        plot_3d_points_crop: bool = False,
        plot_3d_points_plane: bool = False,
        plot_emerging_points: bool = False,
        cluster_threshold: int = 3
    ):
        """
        Plot the agricultural sequence using the Plotly library.

        Args:
            data_plot: a list containing all the previous plotted
                objects. If it is not informed, a empty list is
                created and data is appended to it.
            line_scalars: a Numpy array containing the desired scalars
                to plot the crop line. If it is not informed, the line
                is not plotted.
            plane_scalars: a tuple containing two Numpy arrays
                with scalars to plot the plan. The first Numpy array
                must contain scalars for X coordinates and the second
                must contain scalars for Z coordinates. If it is not
                provided, the plan is not plotted.
            plot_3d_points_crop: a boolean that indicates if the crop 3D
                pointclouds needs to be plotted.
            plot_3d_points_plane: a boolean that indicates if the ground
                plane 3D pointclouds needs to be plotted.
            plot_emerging_point: a boolean that indicates if the crop
                3D emerging point needs to be plotted.
            cluster_threshold: a integer that indicates how many occurences
                a cluster must have to be printed.
        """

        data = []
        if data_plot is not None:
            data = data_plot

        clusters = []
        for scene in self.scenes:
            for crop in scene.crop_group.crops:
                clusters.append(crop.cluster)

        cluster_blacklist = [
            cluster for cluster in clusters if clusters.count(cluster) < cluster_threshold
        ]
        cluster_blacklist.append(-1)

        for scene in self.scenes:
            scene.plot(
                data,
                line_scalars,
                plane_scalars,
                plot_3d_points_crop,
                plot_3d_points_plane,
                plot_emerging_points,
                cluster_blacklist
            )
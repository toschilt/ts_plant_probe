"""
Encapsules several agricultural scenes through time.
"""
import gc
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN

from ts_semantic_feature_detector.features_3d.cluster import Cluster
from ts_semantic_feature_detector.features_3d.scene import AgriculturalScene

class AgriculturalSequence:
    """
    Abstracts a agriculture sequence.

    Attributes:
        scenes (:obj:`list`): a list of :obj:`features_3d.scene.AgriculturalScene` 
            objects containing all the information from each scene in the
            sequence.
        clusters (:obj:`list`): a list of :obj:`features_3d.cluster.Cluster` objects
            containing all the information from each cluster in the sequence.
    """

    def __init__(
        self,
        scenes: List = None
    ):
        """
        Initializes the agricultural sequence.

        Args:
            scenes (:obj:`list`): a list of :obj:`features_3d.scene.AgriculturalScene`
                objects containing all the information from each scene in the sequence.
                If it is None, a empty list is initialzed.
        """
    
        self.scenes = []
        if scenes is not None:
            self.scenes = scenes

        self.clusters = []

    def add_scene(
        self,
        scene: AgriculturalScene,
    ) -> None:
        """
        Adds a scene to this sequence.

        Args:
            scene (:obj:`features_3d.scene.AgriculturalScene`): a scene object
                to be added.
        """
        self.scenes.append(scene)

    def cluster_crops(
        self,
        eps: float = 0.05,
        min_samples: int = 3
    ) -> List:
        """
        Fits a unsupervised model to crop data to try to approximate stems.

        TODO: Save computational power by saving the last seen scene index.

        Args:
            eps (float, optional): the maximum distance between two samples for 
                one to be considered as in the neighborhood of the other.
            min_samples (int, optional): the number of samples (or total weight) 
                in a neighborhood for a point to be considered as a core point.
                This includes the point itself.

        Returns:
            crop_labels (:obj:`list`): containing the labels of the analysed crops.
        """

        descriptors = []
        crops = []

        for scene in self.scenes:
            for crop in scene.crop_group.crops:
                # Checks if the crop is part of a old cluster
                if crop.cluster == -1 or crop.cluster.age != -1:
                    crops.append(crop)
                    emerging_point = crop.emerging_point
                    # angles = crop.crop_vector_angles

                    descriptor = emerging_point[:2]
                    descriptors.append(descriptor)

        if descriptors:
            descriptors = np.array(descriptors)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(descriptors)
            dbscan_clusters = list(dbscan.labels_)

            for old_cluster in self.clusters:
                old_cluster.age += 1

            # Check if there are new clusters
            for nc, dbscan_cluster in enumerate(dbscan_clusters):
                # Check if the new cluster is a outlier
                if dbscan_cluster != -1:
                    if len(self.clusters) < dbscan_cluster + 1:
                        # Create new clusters as needed
                        for _ in range(dbscan_cluster + 1 - len(self.clusters)):
                            self.clusters.append(Cluster(Cluster.next_cluster_id))
                            Cluster.next_cluster_id += 1

                        # Add cluster to crop
                        crops[nc].cluster = self.clusters[-1]
                    else:
                        existing_cluster = self.clusters[dbscan_cluster]

                        # Only reset age for the clusters with crops found in the last scene
                        if nc > len(dbscan_clusters) - len(self.scenes[-1].crop_group.crops):
                            existing_cluster.age = 0

                        # Add cluster to crop
                        crops[nc].cluster = existing_cluster

        print([cluster.id for cluster in self.clusters])
        print([cluster.age for cluster in self.clusters])

    def remove_old_clusters(
        self,
        cluster_max_age: int = 15
    ) -> bool:
        """
        Removes old clusters from the sequence.

        Args:
            cluster_max_age (int, optional): the maximum age of a cluster to be removed.

        Returns:
            bool: True if at least one cluster was removed, False otherwise.
        """
        old_clusters = [[i, cluster] for i, cluster in enumerate(self.clusters) if cluster.age > cluster_max_age]
        
        if old_clusters:
            old_clusters = np.array(old_clusters)
            for c, cluster in zip(old_clusters[:, 0], old_clusters[:, 1]):
                # Indicating that the cluster is expired
                cluster.age = -1

                del self.clusters[c]
                old_clusters[:, 0] -= 1

            return True
        else:
            return False
    
    def get_old_scenes_idxs(
        self,
        old_cluster_exists: bool
    ):
        """
        Gets the scenes that only have crops that are part of old clusters or are outliers.

        Only used when there is at least one old cluster.
        
        Args:
            old_cluster_exists (bool): True if there is at least one old cluster in this scene,
                False otherwise.
        """

        # When there is at least one old cluster, remove scenes with outliers or crops
        # that are part of old clusters.
        old_scenes_idxs = []
        if old_cluster_exists:
            for s, scene in enumerate(self.scenes):
                remove = True
                for crop in scene.crop_group.crops:
                    if crop.cluster != -1 and crop.cluster.age != -1:
                        remove = False
                        break

                if remove:
                    old_scenes_idxs.append(s)

        return old_scenes_idxs
    
    def remove_old_scenes(
        self,
        old_scenes_idxs: List[int]
    ) -> bool:
        """
        Removes old scenes from the sequence.

        Args:
            old_scenes_idxs (list of int): the indexes of the scenes that should be removed.

        Returns:
            bool: True if at least one scene was removed, False otherwise.
        """
        old_scenes_idxs = np.array(old_scenes_idxs)

        for s in old_scenes_idxs:
            # for crop in self.scenes[s].crop_group.crops:
            #     print(gc.get_referrers(crop))

            del self.scenes[s]
            old_scenes_idxs -= 1

        return len(old_scenes_idxs) > 0

    def plot(
        self,
        data_plot: List = None,
        line_scalars: npt.ArrayLike = None,
        plane_scalars: Tuple[npt.ArrayLike, npt.ArrayLike] = None,
        plot_3d_points_crop: bool = False,
        plot_3d_points_plane: bool = False,
        plot_emerging_points: bool = False,
    ) -> List:
        """
        Plot the agricultural sequence using the Plotly library.

        Args:
            data_plot (:obj:`list`, optional): a list containing all
                the previous plotted objects. If it is not informed,
                a empty list is created and data is appended to it.
            line_scalars: (:obj:`np.ndarray`, optional) the desired scalars
                to plot the crop line. If it is not informed, the line
                is not plotted.
            plane_scalars: (tuple of [np.ndarray, np.ndarray], optional):
                the scalars to plot the plan. The first tuple array
                must contain scalars for X coordinates and the second
                must contain scalars for Z coordinates. If it is not
                provided, the plan is not plotted.
            plot_3d_points_crop (bool, optional): indicates if the crop 3D
                pointclouds needs to be plotted.
            plot_3d_points_plane (bool, optional): indicates if the ground
                plane 3D pointclouds needs to be plotted.
            plot_emerging_point (bool, optional): indicates if the crop
                3D emerging point needs to be plotted.

        Returns:
            data_plot (:obj:`list`): a list containing all the plotted objects.
        """

        data = []
        if data_plot is not None:
            data = data_plot

        for scene in self.scenes:
            scene.plot(
                data,
                line_scalars,
                plane_scalars,
                plot_3d_points_crop,
                plot_3d_points_plane,
                plot_emerging_points,
            )

        return data
"""
"""
from typing import Tuple, List

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ts_semantic_feature_detector.features_2d.detections import DetectionGroup
from ts_semantic_feature_detector.features_2d.boxes import Box
from ts_semantic_feature_detector.features_2d.masks import Mask
from ts_semantic_feature_detector.features_3d.camera import StereoCamera
from ts_semantic_feature_detector.features_3d.ground_plane import GroundPlane
from ts_semantic_feature_detector.visualization.colors import get_color_from_cluster

class CornCrop:
    """
    Abstraction of a 3D agricultural corn crop.

    Attributes:
        crop_mask: a features_2d.masks.Mask object containing the 2D
            mask that represents this crop. Used for visualization.
        crop_box: a features_2d.boxes.Box object containing the 2D
            box that represents this crop.
        ps_3d: a Numpy array containing 3D corn crop points.
        average_point: a Numpy array containing the 3D average point
            that describes the crop position.
        crop_vector: a Numpy array containing the 3D vector that
            describes the crop orientation.
        crop_vector_angles: a Numpy array containing the theta and phi
            angles that describes the vector orientation in the 3D space.
        emerging_point: a Numpy array containing the 3D point where
            the crop intercepts the ground plane.
        filter_data: a list containing the histogram and bins values
            obtained from the depth filtering.
        estimated_motion_2d: a tuple containing the 2D estimated motion of
            this crop considering extrinsics information from the moment
            that this is constructed and the next moment.
        cluster: indicates the cluster number that this crop belongs to.
    """

    def __init__(
        self,
        camera: StereoCamera,
        depth_img: Image.Image,
        crop_mask: Mask,
        crop_box: Box,
        depth_neighbors: int = None
    ) -> None:
        """
        Find the 3D corn crop points.

        Construct the 2D points and the masked depth image to
        calculate the 3D points.

        #TODO: visualize data from depth filtering
        #TODO: clean _get_vector_angles method

        Args:
            camera: the features_3d.camera.StereoCamera object. It
                contains all the stereo camera information to obtain
                the 3D crop.
            depth_img: the PIL Image object containing the depth img
                from the whole scene. It will be masked in this method.
            crop_mask: the features_2d.masks.Mask object. It contains
                the crop 2D mask to obtain the 3D crop.
            crop_box: the features_2d.boxes.Box object. It contains
                the crop 2D box to do tracking.
            depth_neighbors: a integer value containing the accepted number of 
                neighbors to the most frequent depth. For more reference, 
                please see documentation for '_filter_crop_depth' method. 
                If it is not provided, the depth is not filtered.
        """
        self.crop_mask = crop_mask
        self.crop_box = crop_box

        # Binary mask indices has data in (height, width) shape
        # Need to swap axis to get analogue x and y indices values
        # in the image frame.
        xy = crop_mask.binary_data_idxs[:, [1, 0]]

        # Insert ones to get 2D points in homogeneous coordinates.
        ps_2d = np.hstack((xy, np.ones((xy.shape[0], 1))))

        # Acess depth_img only in the 2D crop points
        crop_depth = depth_img[
            crop_mask.binary_data_idxs[:, 0],
            crop_mask.binary_data_idxs[:, 1]
        ]
        crop_depth = np.array(crop_depth).astype(float)
        crop_depth /= float(1e3)
        
        # Filter depth to remove outliers.
        if depth_neighbors is not None:
            crop_depth, hist, bins = self._filter_crop_depth(
                crop_depth,
                depth_neighbors)
            self.filter_data = [hist, bins]
        
        # Get the 3D crop points.
        self.ps_3d = []
        for p_2d, z in zip(ps_2d, crop_depth):
            self.ps_3d.append(camera.get_3d_point(p_2d, z))
        self.ps_3d = np.array(self.ps_3d)

        # Get 3D features.
        self.average_point = np.average(self.ps_3d, axis=0)
        self.crop_vector = self._get_principal_component(self.ps_3d)
        self.crop_vector_angles = self._get_vector_angles(self.crop_vector)
        self.emerging_point = None

        # Tracking data
        self.average_depth = self.average_point[2]
        self.estimated_motion_2d = 0
        self.cluster = -1

        # Output data
        self.emerging_point_local_frame = None

    def _filter_crop_depth(
        self,
        masked_depth: Image.Image,
        depth_neighbors: float = 2,
        size_bins: int = 500
    ) -> Tuple[Image.Image, npt.ArrayLike, npt.ArrayLike]:
        """
        Filters the depth image with a distance occurence approach.

        Take the most frequent distance and its 'depth_filter_value' neighbors
        as the depth of the crop. The distance is clipped to the bottom and upper
        neighbors.

        Args:
            masked_depth: the PIL Image object containing the crop masked
                depth information.
            depth_neighbors: a integer value indicating how many neighbors
                will be considered to filter the depth.
            size_bins: a interger value containing the size of the histogram
                bins.

        Returns:
            the PIL Image object containing the filtered depth information.
            a Numpy array containing the histogram values (size i).
            a Numpy array containing the histogram bins values (size i + 1).
        """
        # Get the depth histogram
        hist, bins = np.histogram(
            masked_depth,
            bins=np.linspace(
                np.min(masked_depth),
                np.max(masked_depth),
                size_bins)
        )

        # Find the distance that has the most occurrences
        most_prob_z_bin_idx = np.argmax(hist)

        lowest_bin_idx = most_prob_z_bin_idx - depth_neighbors
        highest_bin_idx = most_prob_z_bin_idx + depth_neighbors

        if lowest_bin_idx < 0:
            lowest_bin_idx = 0

        if highest_bin_idx > bins.shape[0]:
            highest_bin_idx = bins.shape[0] - 1

        lower_z = bins[lowest_bin_idx]
        high_z = bins[highest_bin_idx]

        return np.clip(masked_depth, lower_z, high_z), hist, bins
    
    def _get_principal_component(
        self,
        data_3d: npt.ArrayLike,
    ):
        """
        Get the principal component vector from crop 3D points.

        Args:
            data_3d: the crop's 3D points.
        """
        X = data_3d.reshape(-1, 3)
        pca = PCA(n_components=1)
        pca.fit(X)

        return pca.components_[0]
    
    def _get_vector_angles(
        self,
        vector: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Get theta and phi angles representing vector orientation in 3D space.

        The angle theta is the angle between the Z axis and the vector. The
        angle phi is the angle between the x axis and the projection of the vector
        into the XY plane.

        Args:
            vector: a Numpy array containing the vector coordinates.

        Returns:
            a Numpy array containing the theta and phi values, respectively.
        """
        r = np.linalg.norm(vector, ord=2)
        theta = np.arctan2(vector[2], r)
        phi = np.arctan2(vector[1], vector[0])

        return np.array([theta, phi])
    
    def plot_depth_histogram(
        self,
        rgb_img: npt.ArrayLike,
        depth_img: npt.ArrayLike
    ):
        """
        Plot the crop depth histogram used for filtering.

        Args:
            rgb_img: a Numpy array containing the RGB image. Used
                only for visualization.
            depth_img: a Numpy array containing the depth image. Used
                only for visualization.
        """
        hist = self.filter_data[0]
        bins = self.filter_data[1]

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('Mask analysed')
        plt.imshow(rgb_img)
        self.crop_mask.plot(alpha = 0.7)
        plt.subplot(1, 3, 2)
        plt.title('Depth image')
        plt.imshow(depth_img)
        plt.subplot(1, 3, 3)
        plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")
        plt.xlabel('Depth (mm)')
        plt.ylabel('Occurrences')
        plt.show()

    def find_emerging_point(
        self,
        ground_plane: GroundPlane
    ):
        """
        Finds the crop's emerging point.

        The emerging point is obtained by calculating the interception
        point between the crop and the ground plane.

        Args:
            ground_plane: the features_3d.ground_plane.GroundPlane object.
                It contains all the ground plane features.
        """
        scalar = np.dot(
            ground_plane.average_point - self.average_point,
            ground_plane.normal_vector)
        scalar /= np.dot(ground_plane.normal_vector, self.crop_vector)
        self.emerging_point = self.average_point + scalar*self.crop_vector

        return self.emerging_point
    
    def plot(
        self,
        data_plot: List = None,
        plot_3d_points: bool = False,
        line_scalars: npt.ArrayLike = None,
        plot_emerging_point: bool = False,
        cluster_blacklist: List = None
    ):
        """
        Plot the corn crop using Plotly library.

        Args:
            data_plot: a list containing all the previous plotted
                objects. If it is not informed, a empty list is
                created and data is appended to it.
            plot_3d_points: a boolean that indicates if the crop 3D
                pointcloud needs to be plotted.
            line_scalars: a Numpy array containing the desired scalars
                to plot the crop line. If it is not informed, the line
                is not plotted.
            plot_emerging_point: a boolean that indicates if the crop
                3D emerging point needs to be plotted.
            crop_labels: a list containing the crops' labels.
            cluster_blacklist: a list containing the clusters that must be
                ignored.
        """
    
        data = []
        if data_plot is not None:
            data = data_plot

        color = ''
        if self.cluster not in cluster_blacklist:
            color = get_color_from_cluster(self.cluster)
        else:
            color = '#000000'

        if plot_3d_points:
            data.append(
                go.Scatter3d(
                    x=self.ps_3d[:, 0],
                    y=self.ps_3d[:, 1],
                    z=self.ps_3d[:, 2],
                    marker = go.scatter3d.Marker(
                        size=2, 
                        color=color
                    ),
                    opacity=0.8,
                    mode='markers'
                )
            )

        if line_scalars is not None:
            line_scalars = np.tile(line_scalars, (3, 1)).T
            line = self.average_point + line_scalars*self.crop_vector

            data.append(
                go.Scatter3d(
                    x=line[:, 0],
                    y=line[:, 1],
                    z=line[:, 2],
                    marker = go.scatter3d.Marker(
                        size=2, 
                        color=color
                    ),
                    opacity=0.8,
                    mode='markers'
                )
            )
        
        if plot_emerging_point:
            data.append(
                go.Scatter3d(
                    x=[self.emerging_point[0]],
                    y=[self.emerging_point[1]],
                    z=[self.emerging_point[2]],
                    marker = go.scatter3d.Marker(
                        size=4,
                        color=color
                    ),
                    opacity=0.8,
                    mode='markers'
                )
            )
        
        return data
    
class CornCropGroup:
    """
    Abstraction of a group of 3D agricultural corn crops.

    Attributes:
        crops: a list containing the features_3d.crop.CornCrop objects.
    """

    def __init__(
        self,
        detection_group: DetectionGroup,
        camera: StereoCamera,
        depth_img: Image.Image,
        mask_filter_threshold: float = None,
        ground_plane: GroundPlane = None
    ):
        """
        Initializes a corn crop group.
        
        Args:
            mask_group: a features_2d.detection.DetectionGroup object 
                containing all the crops masks.
            camera: the features_3d.camera.StereoCamera object. It
                    contains all the stereo camera information to obtain
                    the 3D crops.
            depth_img: the PIL Image object containing the depth img
                from the whole scene. It will be masked in this method.
            filter_threshold: a float value containing the threshold to
                filter the depth data. For more reference, please see
                documentation for features_3d.masks.CornCrop._filter_crop_depth method. 
                If it is not provided, the depth is not filtered.
            ground_plane: the features_3d.ground_plane.GroundPlane object.
                It contains all the ground plane features. If it is not
                informed, the crops' emerging point is not calculated.
        """
        
        self.crops = []
        for mask, box in zip(
            detection_group.mask_group.data, 
            detection_group.box_group.data
        ):
            crop = CornCrop(
                    camera,
                    depth_img,
                    mask,
                    box,
                    mask_filter_threshold
                )
            
            if ground_plane is not None:
                crop.find_emerging_point(ground_plane)

                # Creates a copy of the emerging point in the local frame.
                # The original variable can store the emerging point with extrinsic information.
                crop.emerging_point_local_frame = crop.emerging_point

            self.crops.append(crop)

    def plot_depth_histograms(
        self,
        rgb_img: npt.ArrayLike,
        depth_img: npt.ArrayLike
    ):
        """
        Plot the crop depth histograms used for filtering.

        Args:
            rgb_img: a Numpy array containing the RGB image. Used
                only for visualization.
            depth_img: a Numpy array containing the depth image. Used
                only for visualization.
        """
        for crop in self.crops:
            crop.plot_depth_histogram(rgb_img, depth_img)

    def plot(
        self,
        data_plot: List = None,
        plot_3d_points: bool = False,
        line_scalars: npt.ArrayLike = None,
        plot_emerging_point: bool = False,
        cluster_blacklist: List = None
    ):
        """
        Plot the corn group using the Plotly library.

        Args:
            data_plot: a list containing all the previous plotted
                objects. If it is not informed, a empty list is
                created and data is appended to it.
            plot_3d_points: a boolean that indicates if the crop 3D
                pointcloud needs to be plotted.
            line_scalars: a Numpy array containing the desired scalars
                to plot the crop line. If it is not informed, the line
                is not plotted.
            plot_emerging_point: a boolean that indicates if the crop
                3D emerging point needs to be plotted.
            crop_labels: a list containing the crops' labels.
            cluster_blacklist: a list containing the clusters that must be
                ignored.
        """
        data = []
        if data_plot is not None:
            data = data_plot

        for crop in self.crops:
            crop.plot(
                data,
                plot_3d_points,
                line_scalars,
                plot_emerging_point,
                cluster_blacklist
            )

        return data
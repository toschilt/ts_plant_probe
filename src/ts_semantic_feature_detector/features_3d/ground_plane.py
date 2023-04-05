"""
This module contains the GroundPlane class that is used to find the ground
"""

from functools import partial
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ts_semantic_feature_detector.features_3d.camera import StereoCamera

class GroundPlane:
    """
    Abstraction of a 3D agricultural corn ground plane.

    Attributes:
        rgb_img (PIL.Image): the scene RGB image.
        hsv_img (:obj:`np.ndarray`): the scene HSV image.
        binary_mask (:obj:`np.ndarray`): the binary mask.
        ground_vectors (:obj:`np.ndarray`): two vectors that describes
            the ground plane.
        normal_vector (:obj:`np.ndarray`): the normal vector of the
            ground plane.
        coeficients (:obj:`np.ndarray`): the plane equation coeficients.
            The equation used is ax + by + cz + d = 0.
    """

    def __init__(
        self,
        rgb_img: Image.Image,
        finding_ground_method: str,
        camera: StereoCamera,
        depth_img: Image.Image,
        threshold_values: Dict = None,
    ):
        """
        Initializes the ground plane abstraction.

        TODO: Add more documentation about the threshold_values dictionary.
        TODO: refactor this to share some 3D methods with CornCrop

        Args:
            rgb_img (PIL.Image): the scene RGB image.
            finding_ground_method (str): the type of ground finding method 
                that will be used. The possible methods are:

                1. 'threshold_gaussian': converts the RGB to the HSV color 
                space and applies a Gaussian filter and color thresholds
                to find the ground. 

                2. 'ngrdi': uses the NGRDI vegetation index to find the
                ground.
            camera (features_3d.camera.StereoCamera): the object contains all 
                the stereo camera information to obtain the 3D crop.
            depth_img (PIL.Image): the depth img from the whole scene. It will 
                be masked in this method.
            threshold_values (:obj:`dict` [str, float]): the threshold values if 
                the choosen method is 'threshold_gaussian'. If it is not
                provided, default values are used. For more reference,
                please refer to source.
        """
        self.rgb_img = rgb_img

        # Finding the ground mask.
        if finding_ground_method == 'threshold_gaussian':
            if threshold_values is not None:
                self.threshold_values = threshold_values
            else:
                self.threshold_values = {
                    'hLow': 0,
                    'sLow': 0,
                    'vLow': 0,
                    'sHigh': 30,
                    'hHigh': 146,
                    'vHigh': 199,
                    'gaussian_filter': 12}
            
            self.hsv_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_BGR2HSV)
            self.binary_mask = self.get_threshold_gaussian_mask(
                self.hsv_img, 
                self.threshold_values)
        elif finding_ground_method == 'ngrdi':
            self.binary_mask = self.get_ngrdi_mask(
                np.array(self.rgb_img)
            )
            
        self.binary_mask = np.uint8(self.binary_mask != 0)
        self.binary_mask_idxs = np.argwhere(self.binary_mask)

        # Getting ground plane 3D points.
        xy = self.binary_mask_idxs[:, [1, 0]]
        ps_2d = np.hstack((xy, np.ones((xy.shape[0], 1))))

        ground_depth = depth_img[
            self.binary_mask_idxs[:, 0],
            self.binary_mask_idxs[:, 1]
        ]
        ground_depth = np.array(ground_depth).astype(float)
        ground_depth /= float(1e3)

        self.ps_3d = []
        for p_2d, z in zip(ps_2d, ground_depth):
            self.ps_3d.append(camera.get_3d_point(p_2d, z))
        self.ps_3d = np.array(self.ps_3d)

        if self.ps_3d.shape[0] > 0:
            # Find the point and vectors that describe the ground plane.
            self.average_point = np.average(self.ps_3d, axis=0)
            self.ground_vectors = self._get_principal_components(self.ps_3d)
            self.normal_vector = np.cross(
                self.ground_vectors[0],
                self.ground_vectors[1]
            )

            self.coeficients = self._get_plane_coefficients(
                self.normal_vector,
                self.average_point
            )

    def _get_plane_coefficients(
        self,
        normal_vector: npt.ArrayLike,
        point: npt.ArrayLike
    ) -> List:
        """
        Calculate the plane coefficients from the normal vector and a point.

        The plane equation considered is ax + by + cz + d = 0.

        Args:
            normal_vector (:obj:`np.ndarray`): the ground plane normal vector
            point (:obj:`np.ndarray`): a point of the ground plane.

        Returns:
            coefs (:obj:`list`): the plane coefficients in the order: a, b, c, d.
        """
        return [normal_vector[0],
                normal_vector[1],
                normal_vector[2],
                -np.sum(normal_vector*point)]

    def get_ngrdi_mask(
        self,
        rgb_img: npt.ArrayLike    
    ) -> npt.ArrayLike:
        """
        Applies the 'ngrdi' method to find the ground mask.

        Args:
            rgb_img (:obj:`np.ndarray`): the scene image in the RGB 
                color space.

        Returns:
            mask (:obj:`np.ndarray`): the image with the calculated 
                NGRDI metric.
        """

        red_channel = rgb_img[:, :, 0]
        green_channel = rgb_img[:, :, 1]

        return (green_channel - red_channel)/(green_channel + red_channel)

    def get_threshold_gaussian_mask(
        self,
        hsv_img: npt.ArrayLike,
        threshold_values: Dict
    ) -> npt.ArrayLike:
        """
        Applies the 'threshold_gaussian' method to find the ground mask.

        Args:
            hsv_img (:obj:`np.ndarray`): the scene image in the HSV
                color space.
            threshold_values (:obj:`dict` [str, float]]): the threshold values if 
                the choosen method is 'threshold_gaussian'. If it is not
                provided, default values are used. For more reference,
                please refer to source.

        Returns:
            mask (:obj:`np.ndarray`): the mask with the ground pixels.
        """
        gaussian_filter = (
            2*threshold_values['gaussian_filter'] + 1,
            2*threshold_values['gaussian_filter'] + 1)
        
        gaussian_img = cv2.GaussianBlur(
            hsv_img,
            gaussian_filter,
            0)

        lower_color_bounds = np.array(
            (threshold_values['hLow'], 
             threshold_values['sLow'],
             threshold_values['vLow'])
        )
        higher_color_bounds = np.array(
            (threshold_values['hHigh'],
             threshold_values['sHigh'],
             threshold_values['vHigh'])
        )
        
        mask = cv2.inRange(
            gaussian_img,
            lower_color_bounds,
            higher_color_bounds)

        return mask

    def _parameter_trackbars_callback(
        self,
        name: str,
        val: int
    ) -> None:
        """
        Callback function to tune color threshold filter size.

        Args:
            name (str): the parameter's name.
            val (int): the parameter's value.
        """
        self.threshold_values[name] = val

    def tune_values_tool(
        self
    ) -> None:
        """
        Implements a tool to tune values for ground plane finding.

        Uses the OpenCV library to do filtering and to open trackbars.
        The 'threshold_gaussian' option needs to be specified in the
        class constructor.
        """
        cv2.namedWindow('control', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        for key in self.threshold_values.keys():
            cv2.createTrackbar(
                key,
                'control',
                self.threshold_values[key],
                255,
                partial(self._parameter_trackbars_callback, key)
            )
        
        cv2.imshow('original', self.rgb_img)

        while True:
            self.binary_mask = self.get_threshold_gaussian_mask(
                self.hsv_img,
                self.threshold_values
            )

            cv2.imshow('image', self.binary_mask)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.binary_mask = np.uint8(self.binary_mask != 0)
        cv2.destroyAllWindows()

    def _get_principal_components(
        self,
        data_3d: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Get the two principal component vector from crop 3D points.

        Args:
            data_3d (:obj:`np.ndarray`): the crop's 3D points.

        Returns:
            components (:obj:`np.ndarray`): the two principal components.
        """
        X = data_3d.reshape(-1, 3)
        pca = PCA(n_components=2)
        pca.fit(X)

        return pca.components_[:2]
    
    def evaluate_at_determined_x_z(
        self,
        x: npt.ArrayLike,
        z: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Evaluate plane Y coordinates at desired X and Z coordinates.

        Args:
            x (:obj:`np.ndarray`): the desired X coordinates.
            z (:obj:`np.ndarray`): the desired Z coordinates.

        Returns:
            x (:obj:`np.ndarray`): x coordinates.
            y (:obj:`np.ndarray`): y coordinates.
            z (:obj:`np.ndarray`): z coordinates.
        """
        x, z = np.meshgrid(x, z)
        y = -(+ self.coeficients[0]*x
              + self.coeficients[2]*z 
              + self.coeficients[3])/self.coeficients[1]
        return x, y, z

    def plot(
        self,   
        data_plot: List = None,
        plot_3d_points: bool = False,
        plot_plan_scalars: Tuple[npt.ArrayLike, npt.ArrayLike] = None
    ) -> List:
        """
        Plot the corn crop using Plotly library.

        Args:
            data_plot (:obj:`list`, optional): the previous plotted objects. 
                If it is not informed, a empty list is created and data
                is appended to it.
            plot_3d_points (bool, optional): a boolean that indicates if 
                the ground plane 3D pointcloud needs to be plotted.
            plot_plan_scalars (a tuple [:obj:`np.ndarray`, :obj:`np.ndarray`], optional):
                two Numpy arrays with scalars to plot the plan. The first Numpy 
                array must contain scalars for X coordinates and the second must
                contain scalars for Z coordinates. If it is not provided, the 
                plan is not plotted.

        Returns:
            data (:obj:`list`): the list of plotted objects.
        """
        data = []
        if data_plot is not None:
            data = data_plot

        if plot_3d_points:
            data.append(
                go.Scatter3d(
                    x=self.ps_3d[:, 0],
                    y=self.ps_3d[:, 1],
                    z=self.ps_3d[:, 2],
                    marker = go.scatter3d.Marker(size=2),
                    opacity=0.8,
                    mode='markers'
                )
            )

        if plot_plan_scalars is not None:
            x, y, z = self.evaluate_at_determined_x_z(
                plot_plan_scalars[0],
                plot_plan_scalars[1]
            )

            data.append(
                go.Surface(
                x=x,
                y=y,
                z=z
                )
            )

        return data
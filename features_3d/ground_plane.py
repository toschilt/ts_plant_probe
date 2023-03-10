"""
"""
from functools import partial
from typing import Dict

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.decomposition import PCA

from features_3d.camera import StereoCamera

class GroundPlane:
    """
    Abstraction of a 3D agricultural corn ground plane.

    Attributes:
        rgb_img: a PIL Image object containing the scene RGB image.
        hsv_img: a Numpy array object containing the scene HSV image.
        binary_mask: a Numpy array containing the binary mask 
    """

    def __init__(
        self,
        rgb_img: Image.Image,
        finding_ground_method: str,
        camera: StereoCamera,
        depth_img: Image.Image,
        threshold_values: Dict = None,
    ) -> None:
        """
        Initializes the ground plane abstraction.

        #TODO: Add more documentation about the threshold_values dictionary.
        #TODO: Implement 'ngrdi' method.
        #TODO: refactor this to share some 3D methods with CornCrop

        Args:
            rgb_img: a PIL Image object containing the scene RGB image.
            finding_ground_method: a string containing the type of ground
                finding method that will be used. The possible methods are:
                1. 'threshold_gaussian': converts the RGB to the HSV color
                    space and applies a Gaussian filter and color thresholds
                    to find the ground. 
                2. 'ngrdi': uses the NGRDI vegetation index to find the
                    ground.
            camera: the features_3d.camera.StereoCamera object. It
                contains all the stereo camera information to obtain
                the 3D crop.
            depth_img: the PIL Image object containing the depth img
                from the whole scene. It will be masked in this method.
            threshold_values: a dictionary containing threshold values if 
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
            self.binary_mask = np.uint8(self.binary_mask != 0)
        elif finding_ground_method == 'ngrdi':
            self.binary_mask = None

        self.binary_mask_idxs = np.argwhere(self.binary_mask)

        # Getting ground plane 3D points.
        xy = self.binary_mask_idxs[:, [1, 0]]
        ps_2d = np.hstack((xy, np.ones((xy.shape[0], 1))))

        ground_depth = depth_img[
            self.binary_mask_idxs[:, 0],
            self.binary_mask_idxs[:, 1]
        ]

        self.ps_3d = []
        for p_2d, z in zip(ps_2d, ground_depth):
            self.ps_3d.append(camera.get_3d_point(p_2d, z))
        self.ps_3d = np.array(self.ps_3d)

        # Find the point and vectors that describe the ground plane.
        self.average_point = np.average(self.ps_3d, axis=0)
        self.ground_vectors = self._get_principal_components(self.ps_3d)

    def get_threshold_gaussian_mask(
        self,
        hsv_img: npt.ArrayLike,
        threshold_values: Dict
    ):
        """
        Applies the 'threshold_gaussian' method to find the ground mask.

        Args:
            hsv_img: a Numpy array containing the scene image in 
                the HSV color space.
            threshold_values: a dictionary containing threshold values if 
                the choosen method is 'threshold_gaussian'. If it is not
                provided, default values are used. For more reference,
                please refer to source.
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
        val: int):
        """
        Callback function to tune color threshold filter size.

        Args:
            name: a string containing the parameter's name
            val: a integer value containing the parameter's value
        """
        self.threshold_values[name] = val

    def tune_values_tool(self):
        """
        Implements a tool to tune values for ground plane finding.

        Uses the OpenCV library to do filtering and to open trackbars.
        The 'threshold_gaussian' option needs to be specified in the
        class constructor.
        """
        cv2.namedWindow('control', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        for key in self.threshold_values.keys():
            cv2.createTrackbar(
                key,
                'control',
                self.threshold_values[key],
                255,
                partial(self._parameter_trackbars_callback, key)
            )

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
    ):
        """
        Get the two principal component vector from crop 3D points.

        Args:
            data_3d: the crop's 3D points.
        """
        X = data_3d.reshape(-1, 3)
        pca = PCA(n_components=2)
        pca.fit(X)

        return pca.components_[:2]
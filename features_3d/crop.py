"""
"""
from typing import Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.decomposition import PCA

from features_2d.masks import Mask
from features_3d.camera import StereoCamera

class CornCrop:
    """
    Abstraction of a 3D agricultural corn crop.

    Attributes:
        ps_3d: a Numpy array containing 3D corn crop points.
        average_point: a Numpy array containing the 3D average point
            that describes the crop position.
        crop_vector: a Numpy array containing the 3D vector that
            describes the crop orientation. 
    """

    def __init__(
        self,
        camera: StereoCamera,
        depth_img: Image.Image,
        crop_mask: Mask,
        filter_threshold: float = None
    ) -> None:
        """
        Find the 3D corn crop points.

        Construct the 2D points and the masked depth image to
        calculate the 3D points.

        #TODO: visualize data from depth filtering

        Args:
            camera: the features_3d.camera.StereoCamera object. It
                contains all the stereo camera information to obtain
                the 3D crop.
            depth_img: the PIL Image object containing the depth img
                from the whole scene. It will be masked in this method.
            crop_mask: the features_2d.masks.Mask object. It contains
                all the 2D crop information to obtain the 3D crop.
            filter_threshold: a float value containing the threshold to
                filter the depth data. For more reference, please see
                documentation for '_filter_crop_depth' method. If it is
                not provided, the depth is not filtered.
        """

        # Binary mask indices has data in (height, width) shape
        # Need to swap axis to get analogue x and y indices values
        # in the image frame.
        xy = crop_mask.binary_data_idxs[:, [1, 0]]

        # Insert ones to get 2D points in homogeneous coordinates.
        ps_2d = np.hstack((xy, np.ones((xy.shape[0], 1))))

        # Acess depth_img only in the 2D crop points
        crop_depth = depth_img[
            crop_mask.binary_data_idxs[:, 0],
            crop_mask.binary_data_idxs[:, 1]]
        
        if filter_threshold is not None:
            crop_depth, hist, bins = self._filter_crop_depth(
                crop_depth,
                filter_threshold)
        
        self.ps_3d = []
        for p_2d, z in zip(ps_2d, crop_depth):
            self.ps_3d.append(camera.get_3d_point(p_2d, z))
        self.ps_3d = np.array(self.ps_3d)

        self.average_point = np.average(self.ps_3d, axis=0)
        self.crop_vector = self._get_principal_component(self.ps_3d)

    def _filter_crop_depth(
        self,
        masked_depth: Image.Image,
        hist_derivative_threshold: float = 70,
        size_bins: int = 100
    ) -> Tuple[Image.Image, npt.ArrayLike, npt.ArrayLike]:
        """
        Filters the depth image with a distance occurence approach.

        This method calculates the histogram of the image and tries to
        isolate the closest distances from the most-occurring-distance
        by thresholding the difference between consecutive distance 
        occurrences values.

        Args:
            masked_depth: the PIL Image object containing the crop masked
                depth information.
            hist_derivative_threshold: a float value to be applied as
                threshold when scanning the difference between consecutive
                distance occurrences values.
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

        # Compute histogram derivative and find the points where it is above
        # the specified threshold
        hist_derivative = np.abs(hist[1:] - hist[:-1])
        filtered_hist_derivative = hist_derivative > hist_derivative_threshold
        
        # Find the extremities of the occurence distribution
        rising_idx = np.where(~filtered_hist_derivative & np.roll(filtered_hist_derivative,-1))[0]
        rising_idx = rising_idx[rising_idx <= most_prob_z_bin_idx]
        falling_idx = np.where(~np.roll(filtered_hist_derivative,-1) & filtered_hist_derivative)[0]
        falling_idx = falling_idx[falling_idx >= most_prob_z_bin_idx]
        
        if rising_idx.any():
            lower_idx_value = rising_idx[-1]
        else:
            lower_idx_value = 0

        if falling_idx.any():
            higher_idx_value = falling_idx[0]
        else:
            higher_idx_value = len(bins) - 3

        # Use the depth at the extremities to clip the depth values
        lower_z = bins[lower_idx_value + 2]
        high_z = bins[higher_idx_value + 2]
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
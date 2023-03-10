"""
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image

class StereoCamera:
    """
    Abstracts some properties and methods related to the StereoCamera.

    Attributes:
        intrinsics_matrix - a 3x3 Numpy array containing the instrinsics 
            matrix values (fx, fy, cx, cy).
        intrinsics_matrix_inv - a 3x3 Numpy array containing the inverse
            intrinsics matrix.
    """

    def __init__(
        self,
        intrinsics: Tuple[int, int, int, int],
    ) -> None:
        """
        Initializes the stereo camera.

        Args:
            intrinsecs - a tuple containing the four stereo camera
                intrinsics values ([fx, fy, cx, cy]).
        """

        self.intrinsics_matrix = np.array(
            [[intrinsics[0], 0, intrinsics[2]],
             [0, intrinsics[1], intrinsics[3]],
             [0, 0, 1]])
        self.intrinsics_matrix_inv = np.linalg.inv(self.intrinsics_matrix)

    def filter_depth(
        self,
        depth_img: Image.Image,
        hist_derivative_threshold: int = 70,
        size_bins: int = 100
    ) -> Tuple[Image.Image, npt.ArrayLike, npt.ArrayLike]:
        """
        Filters the depth image with a distance occurence approach.

        This method calculates the histogram of the image and tries to
        isolate the closest distances from the most-occurring-distance
        by thresholding the difference between consecutive distance 
        occurrences values.

        Args:
            depth_img: the PIL Image object containing the depth information.
            hist_derivative_threshold: a integer value to be applied as
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
            depth_img,
            bins=np.linspace(
                np.min(depth_img),
                np.max(depth_img),
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
        return np.clip(depth_img, lower_z, high_z), hist, bins
    
    def get_3d_point(
        self,
        p_2d: npt.ArrayLike,
        depth: float,
    ):
        """
        Gets the 3D points in the camera reference frame.

        Uses the equation p_3d = z*(K_inv @ p_2d), where p_3d is the
        3D point vector, z is the depth information, K_inv the inverse
        intrinsics matrix and p_2d is the 2D point vector (in homogeneous
        coordinates). p_2d is in homogeneous coordinates, but p_3d is not. 
        The scalar z makes the operation correct.

        Args:
            p_2d: the 2D point in homogeneous coordinates ([x, y, 1]).
            depth: the depth information related to the 2D point.
        """
        return depth*(self.intrinsics_matrix_inv @ p_2d)

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
        size - a tuple containing the image size (width, height)
    """

    def __init__(
        self,
        intrinsics: Tuple[int, int, int, int],
        size: Tuple[int, int]
    ) -> None:
        """
        Initializes the stereo camera.

        Args:
            intrinsecs - a tuple containing the four stereo camera
                intrinsics values ([fx, fy, cx, cy]).
            size - a tuple containing the image size (width, height)
        """

        self.intrinsics_matrix = np.array(
            [[intrinsics[0], 0, intrinsics[2]],
             [0, intrinsics[1], intrinsics[3]],
             [0, 0, 1]])
        self.intrinsics_matrix_inv = np.linalg.inv(self.intrinsics_matrix)
        self.size = np.array(size)
    
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

    def get_2d_point(
        self,
        p_3d: npt.ArrayLike
    ):
        """
        Gets the 2D point in the image reference frame.

        Uses the equation p_2d = 1/z*(K @ p_3d), where p_2d is the
        2D point vector (in homogenenous coordinates), z is the depth
        information, K is the intrinsics matrix and p_3d is the 3D point
        vector. p_2d is in homogeneous coordinates, but p_3d is not. 
        The scalar 1/z makes the operation correct.

        The depth information can be ignored, as the z coordinate can be
        equivalently assumed to be 1 as the equation above shows.

        Args:
            p_3d: the 3D point in homogeneous coordinates ([x, y, z, w]).
        """

        intrinsics = np.append(self.intrinsics_matrix, [[0], [0], [0]], axis=1)
        p_2d = intrinsics @ p_3d
        return p_2d[:2] / p_2d[2]
    
    def load_image(
        self,
        image_path: str,
    ) -> npt.ArrayLike:
        """
        Loads a single image using the PIL library.

        Has the purpose of avoiding explicit loading in main programs.

        Args:
            image_path: a string containing the path to the
                image.
        
        Returns:
            a Numpy array containing the data from hte image.
        """
        return np.array(Image.open(image_path))
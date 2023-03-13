"""
"""
import typing as List

import numpy as np
import numpy.typing as npt

from ts_semantic_feature_detector.features_3d.crop import CornCropGroup
from ts_semantic_feature_detector.features_3d.ground_plane import GroundPlane

class AgriculturalScene:
    """
    Abstracts a agriculture scene.

    A agricultural scene containing 3D crops and ground plane. It is
    obtained from a RGB and a depth images.

    Attributes:
        crop_group: a features_3d.crop.CornCropGroup object. It encapsules
            the information about all the crops in a single scene.
        ground_plane: the features_3d.ground_plane.GroundPlane object.
            It contains all the ground plane features.
        extrinsics: a Numpy array containing the extrinsics matrix.
            It can be applied to all agricultural scene components. If it
            is not informed, the add_extrinsics_information function must
            be called.
    """

    def __init__(
        self,
        crop_group: CornCropGroup,
        ground_plane: GroundPlane,
        extrinsics: npt.ArrayLike = None
    ):
        self.crop_group = crop_group
        self.ground_plane = ground_plane
        self.extrinsics = extrinsics

    def _apply_extrinsics_to_3D_vector(
        self,
        vector_3d: npt.ArrayLike,
        extrinsics: npt.ArrayLike
    ):
        """
        Applies the extrinsics matrix to a 3D vector.

        Args:
            vector_3d: a Numpy array (3x1) containing the 3D vector. It
                will be transformed in homogeneous coordinate to apply
                the extrinsics.
            extrinsics: a Numpy array (4x4) with the extrinsics matrix.
                It describes the transformation from the camera frame to
                a global frame.

        Returns:
            a Numpy array (3x1) containing the 3D vector in Euclidian
            coordinates.
        """
        ext_hom_3d = extrinsics @ np.append(vector_3d, 1)
        return ext_hom_3d[:-1]/ext_hom_3d[-1]
        

    def add_extrinsics_information(
        self,
        translation: List,
        rotation: List
    ):
        """
        Adds extrinsics information to the scene.
        
        Updates the crops and ground plane 3D points and their's describing
        features (average points and vectors).

        Args:
            translation: a tuple containing three floats describing the
                desired translation in the x, y and z axis, respectively.
            rotation: a tuple containing three floats describing the 
                desired yaw, pitch and roll rotations.
        """
        if self.extrinsics is None:
            # Get the extrinsics matrix.
            R_yaw = np.array(
                [[np.cos(rotation[0]), -np.sin(rotation[0]), 0],
                [np.sin(rotation[0]), np.cos(rotation[0]), 0],
                [0, 0, 1]])
            R_pitch = np.array(
                [[np.cos(rotation[1]), 0, np.sin(rotation[1])],
                [0, 1, 0],
                [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])
            R_roll = np.array(
                [[1, 0, 0],
                [0, np.cos(rotation[2]), -np.sin(rotation[2])],
                [0, np.sin(rotation[2]), np.cos(rotation[2])]])
            R = R_yaw @ R_pitch @ R_roll
            t = np.array([translation[0], translation[1], translation[2]])[:, None]
            self.extrinsics = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))

        # Modfying the crops
        for crop in self.crop_group.crops:
            for p_3d in crop.ps_3d:
                p_3d = self._apply_extrinsics_to_3D_vector(p_3d, self.extrinsics)

            crop.average_point = self._apply_extrinsics_to_3D_vector(crop.average_point, self.extrinsics)
            crop.crop_vector = self._apply_extrinsics_to_3D_vector(crop.crop_vector, self.extrinsics)
            crop.emerging_point = self._apply_extrinsics_to_3D_vector(crop.emerging_point, self.extrinsics)

        # Modfying the ground plane
        for p_3d in self.ground_plane.ps_3d:
            p_3d = self._apply_extrinsics_to_3D_vector(p_3d, self.extrinsics)

        for vector in self.ground_plane.ground_vectors:
            vector = self._apply_extrinsics_to_3D_vector(vector, self.extrinsics)

        self.ground_plane.normal_vector = self._apply_extrinsics_to_3D_vector(
            self.ground_plane.normal_vector,
            self.extrinsics)
        
        self.ground_plane.average_point = self._apply_extrinsics_to_3D_vector(
            self.ground_plane.average_point,
            self.extrinsics
        )
        
        self.ground_plane.coeficients = self.ground_plane._get_plane_coefficients(
            self.ground_plane.normal_vector,
            self.ground_plane.average_point
        )
        
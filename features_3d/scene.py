"""
"""
from typing import List, Tuple

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

        #TODO: crop and ground plane calculation are done before and after
            adding extrinsics. Refactor constructors to spare computational
            power.
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

        # Modfying the ground plane
        for p_3d in self.ground_plane.ps_3d:
            p_3d = self._apply_extrinsics_to_3D_vector(p_3d, self.extrinsics)

        self.ground_plane.average_point = np.average(
            self.ground_plane.ps_3d,
            axis=0
        )
        self.ground_plane.ground_vectors = self.ground_plane._get_principal_components(
            self.ground_plane.ps_3d
        )
        self.ground_plane.normal_vector = np.cross(
            self.ground_plane.ground_vectors[0],
            self.ground_plane.ground_vectors[1]
        )
        self.ground_plane.coeficients = self.ground_plane._get_plane_coefficients(
            self.ground_plane.normal_vector,
            self.ground_plane.average_point
        )

        # Modfying the crops
        for crop in self.crop_group.crops:
            for p_3d in crop.ps_3d:
                p_3d = self._apply_extrinsics_to_3D_vector(p_3d, self.extrinsics)

            crop.average_point = np.average(crop.ps_3d, axis=0)
            crop.crop_vector = crop._get_principal_component(crop.ps_3d)
            crop.emerging_point = crop.find_emerging_point(
                self.ground_plane
            )

    def plot(
        self,
        data_plot: List = None,
        line_scalars: npt.ArrayLike = None,
        plane_scalars: Tuple[npt.ArrayLike, npt.ArrayLike] = None,
        plot_3d_points_crop: bool = False,
        plot_3d_points_plane: bool = False,
        plot_emerging_points: bool = False
    ):
        """
        Plot the agricultural scene using the Plotly library.

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
        """

        data = []
        if data_plot is not None:
            data = data_plot

        self.crop_group.plot(
            data,
            plot_3d_points_crop,
            line_scalars,
            plot_emerging_points
        )
        
        self.ground_plane.plot(
            data,
            plot_3d_points_plane,
            plane_scalars
        )

        return data
"""
"""

from typing import List, Tuple

import numpy.typing as npt

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
        scene: AgriculturalScene
    ):
        """
        Adds a scene to this sequence.

        Args:
            scene: a features_3d.scene.AgriculturalScene object to be
            added.
        """
        self.scenes.append(scene)

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
                plot_emerging_points
            )
"""
"""

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
        ground_plane: ground_plane: the features_3d.ground_plane.GroundPlane object.
            It contains all the ground plane features.
    """

    def __init__(
        self,
        crop_group: CornCropGroup,
        ground_plane: GroundPlane
    ):
        self.crop_group = crop_group
        self.ground_plane = ground_plane
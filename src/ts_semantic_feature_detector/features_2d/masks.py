"""
Implements all the functionality related to 2D masks.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn import linear_model

from ts_semantic_feature_detector.features_2d.curves import Curve2D
from ts_semantic_feature_detector.features_2d.curves import Line2D

class Mask:
    """
    Contains a single mask.

    Implements methods to process mask and visualize data.

    Attributes:
        data (:obj:`np.ndarray`): the mask data with shape
            (height, weight, color_channel).
        binary_data (:obj:`np.ndarray`): the binary mask data with shape
            (height, weight).
        binary_data_idxs (:obj:`np.ndarray`): the indices where binary_data
            attribute is not False. It is useful to calculate mask curves.
        binary_threshold (float): the threshold with the mask is binarized.
        average_curve (:obj:`features2d.curves.Curve2D`): it describes the 
            curve positioned in the middle portion of the mask.
        ransac_line (:obj:`features2d.curves.Line2D`): it describes the
            fitted line with average_curve points with RANSAC.
        x_bottom (float): the X coordinate evaluated at the mask last row. 
            It is used for redundancy filtering.
    """

    def __init__(
        self,
        mask: npt.ArrayLike,
        binary_threshold: float,
    ):
        """
        Initialize a single mask.

        Args:
            mask (:obj:`np.ndarray`): the mask data with shape
                (color_channel, height, width).
            binary_threshold (float): the threshold value that this mask 
                will be binarized. Binary masks are extensively used
                in this project, so this argument is mandatory.
        """
        self.data = np.moveaxis(mask, 0, 2)
        self.binary_data = np.uint8(self.data > binary_threshold)[:, :, 0]
        self.binary_data_idxs = np.argwhere(self.binary_data)

        self.binary_threshold = binary_threshold

        self.average_curve = None
        self.ransac_line = None
        self.x_bottom = None

    def _get_average_curve(
        self,
        binary_data_indxs: npt.ArrayLike,
    ) -> Curve2D:
        """
        Get the "average" curve from the mask.

        For each unique Y coordinate from the binary mask, it averages
        the X coordinates.

        Args:
            binary_data_indxs (:obj:`np.ndarray`): the indices where binary_data
                attribute is not False.
        
        Returns:
            avg_curve (:obj:`features2d.curves.Curve2D`): average curve object.
        """
        sorted_idxs = np.lexsort(binary_data_indxs[:, [1, 0]].T)
        sorted_xy = binary_data_indxs[sorted_idxs, :]
        unique_y, unique_y_idx = np.unique(sorted_xy[:, 0], return_index=True)
        x_values = np.split(binary_data_indxs[:, 1], unique_y_idx[1:])

        average_x = []
        for x in x_values:
            average_x.append(np.average(x))
        average_x = np.array(average_x)

        return Curve2D(average_x, unique_y)
    
    def _get_RANSAC_line(
        self,
        average_curve: Curve2D,
    ) -> Line2D:
        """
        Fit a line to the average mask points with RANSAC.

        Args:
            average_curve (:obj:`features2d.curves.Curve2D`): the curve 
                positioned in the middle portion of the mask.
        
        Returns:
            ransac_line (:obj:`features2d.curves.Line2D`): RANSAC line object.
        """
        ransac = linear_model.RANSACRegressor(min_samples=2)
        X = average_curve.x.reshape(-1, 1)
        y = average_curve.y.reshape(-1, 1)
        ransac = ransac.fit(X, y)
        angular_coef = ransac.estimator_.coef_[0][0]
        linear_coef = ransac.estimator_.intercept_[0]

        scalars = np.linspace(
            np.min(average_curve.y),
            np.max(average_curve.y)
        )

        return Line2D(angular_coef, linear_coef, y=scalars)

    def extract_curves(
        self
    ) -> None:
        """
        Extract curves from the mask to do posterior filtering.

        Storage the curves in average_curve and ransac_line attributes.
        
        This algorithm is a two step process:

        1. Gets a curve whose X coordinate points are the average X
        coordinates from the mask for each unique Y coordinate also 
        in the mask.
        
        2. Fits a line through the 'average points' using RANSAC.
        """
        self.average_curve = self._get_average_curve(self.binary_data_idxs)
        self.ransac_line = self._get_RANSAC_line(self.average_curve)

    def plot(
        self,
        alpha: float = 1.0
    ) -> None:
        """
        Plot a single mask using the Matplolib library.

        Args:
            alpha (float): the mask transparency amount.
        """
        plt.imshow(np.ma.masked_where(self.binary_data == 0, self.binary_data), alpha=alpha)

class MaskGroup:
    """
    Agroup masks to do filtering.

    Implements filtering using inference metrics and comparing data 
    from different masks.

    Attributes:
        data (:obj:`list`): the masks as :obj:`features_2d.masks.Mask` 
            objects.
    """

    def __init__(
        self,
        masks: npt.ArrayLike,
        binary_threshold: float,
    ):
        """
        Initializes the group of masks.

        Args:
            masks (:obj:`np.ndarray`): the masks with shape
                (num_masks, color_channel, height, width). This is the same
                format outputted by the Mask RCNN network.
            binary_threshold (float): the threshold with the masks will be 
                binarized. Binary masks are extensively used in this project, 
                so this argument is mandatory.
        """
        self.data = []
        for mask in masks:
            self.data.append(Mask(mask, binary_threshold))

    def extract_curves(
        self
    ) -> None:
        """
        Extract the curves from all the masks in the group.

        For more information about how the curves are obtained,
        please refer to the :method:`Mask.extract_curves` method 
        documentation.
        """
        for mask in self.data:
            mask.extract_curves()

    def plot(
        self,
        alpha: float = 1.0
    ) -> None:
        """
        Plot the group of masks using Matplolib library.

        Args:
            alpha (float): the mask transparency amount.
        """
        for mask in self.data:
            mask.plot(alpha)
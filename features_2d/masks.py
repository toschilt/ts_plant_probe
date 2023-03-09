"""
"""

import numpy as np
import numpy.typing as npt
from sklearn import linear_model

from features_2d.curves import Curve2D
from features_2d.curves import Line2D

class Mask:
    """
    Contains a single mask.

    Implements methods to process mask and visualize data.

    Attributes:
        data - a Numpy array containing the mask data with shape
            (height, weight, color_channel)
        binary_data - a Numpy array containing the binary mask data
            with shape (height, weight)
        binary_data_idx - a Numpy array containing all the indices
            where binary_data attribute is not False. It is useful
            to calculate mask curves.
        average_curve - a features2d.curves.Curve2D object. It describes
            the curve positioned in the middle portion of the mask.
        ransac_line - a features2d.curves.Line2D object. It describes the
            fitted line with average_curve points with RANSAC.
    """

    def __init__(
        self,
        mask: npt.ArrayLike,
        binary_threshold: float,
    ):
        """
        Initialize a single mask.

        Args:
            mask - a Numpy array containing the mask data with shape
                (color_channel, height, width)
            binary_threshold - a float number representing the threshold with
                this mask will be binarized. Binary masks are extensively used
                in this project, so this argument is mandatory.
        """
        self.data = np.moveaxis(mask, 0, 2)
        self.binary_data = np.uint8(self.data > binary_threshold)[:, :, 0]
        self.binary_data_idxs = np.argwhere(self.binary_data)

        self.average_curve = None
        self.ransac_line = None

    def _get_average_curve(
        self,
        binary_data_indxs: npt.ArrayLike,
    ) -> Curve2D:
        """
        Get the "average" curve from the mask.

        For each unique Y coordinate from the binary mask, it averages
        the X coordinates.

        Args:
            binary_data_indxs: a Numpy array containing all the indices
            where binary_data attribute is not False.
        
        Returns:
            a features2d.curves.Curve2D object.
        """
        
        sorted_idxs = np.lexsort(binary_data_indxs[:, [1, 0]].T)
        sorted_xy = binary_data_indxs[sorted_idxs, :]
        unique_y, unique_y_idx = np.unique(sorted_xy[:, 0], return_index=True)
        x_values = np.split(binary_data_indxs[:, 1], unique_y_idx[1:])

        for i, x in zip(range(0, len(x_values)), x_values):
            x_values[i] = np.average(x)
        average_x = np.array(x_values)

        return Curve2D(average_x, unique_y)
    
    def _get_RANSAC_line(
        self,
        average_curve: Curve2D,
    ) -> Line2D:
        """
        Fit a line to the average mask points with RANSAC

        Args:
            average_curve: a features2d.curves.Curve2D object. It describes
            the curve positioned in the middle portion of the mask.
        
        Returns:
            a features2d.curves.Line2D object.
        """
        ransac = linear_model.RANSACRegressor(min_samples=2)
        X = average_curve.x.reshape(-1, 1)
        y = average_curve.y.reshape(-1, 1)
        ransac = ransac.fit(X, y)
        angular_coef = ransac.estimator_.coef_[0][0]
        linear_coef = ransac.estimator_.intercept_[0]

        return Line2D(angular_coef, linear_coef)

    def extract_curves(self):
        """
        Extract curves from the mask to do posterior filtering.

        Storage the curves in average_curve and ransac_line attributes.
        
        This algorithm is a two step process:
            1. Gets a curve whose X coordinate points are the average X
                coordinates from the mask for each unique Y coordinate 
                also in the mask.
            2. Fits a line through the 'average points' using RANSAC.
        """
        self.average_curve = self._get_average_curve(self.binary_data_idxs)
        self.ransac_line = self._get_RANSAC_line(self.average_curve)

class MaskGroup:
    """
    Agroup masks to do filtering

    Implements filtering using inference metrics and comparing data 
    from different masks.

    Attributes:
        masks - a list containing all the masks as features_2d.masks.Mask
            objects. 
    """

    def __init__(
        self,
        masks: npt.ArrayLike,
        scores: npt.ArrayLike,
        binary_threshold: float,
    ):
        """
        Initializes the group of masks.

        Args:
            masks - a Numpy array containing the masks with shape
                (num_masks, color_channel, height, width). This is the same
                format outputted by the Mask RCNN network.
            scores - a Numpy array containing the masks scores provided by
                the network. Masks and scores must be at the same corresponding
                order.
            binary_threshold - a float number representing the threshold with
                the masks will be binarized. Binary masks are extensively used
                in this project, so this argument is mandatory.
        """

        self.masks = []
        for mask in masks:
            self.masks.append(Mask(mask, binary_threshold))
        self.scores = scores

    def metric_filtering(
        self,
        type: str,
        score_threshold: float = 0.5,
        percentage: float = 0.5,
    ):
        """
        Filter the mask group by some metric.

        It discards the masks and scores considered worst.

        Args:
            type: a string containing the type of metric that will be used to 
                filter the group. It can be 'score' or 'percentage'. The first
                one will filter the group by an arbitrary score threshold value.
                The second will filter the group to remain only the best provided
                percentage of masks.
            score_threshold: a float value at interval [0, 1]. Masks with scores
                below this threshold are deleted.
            percentage: a float value at interval [0, 1]. It determines the
                percentage of best masks that are desired.
        """
        if type == 'score':
            idxs = np.where(self.scores < score_threshold)[0]
            self.scores = np.delete(self.scores, idxs)
            for idx in idxs:
                del self.masks[idx]
        elif type == 'percentage':
            num_predictions = int(len(self.masks)*percentage)
            del self.masks[num_predictions:]
            np.delete(self.scores, range(num_predictions, len(self.scores)))

    def extract_curves(self):
        """
        Extract the curves from all the masks in the group.

        For more information about how the curves are obtained,
        please refer to the Mask.extract_curves method documentation.
        """
        for mask in self.masks:
            mask.extract_curves()


"""
"""
import numpy as np
import numpy.typing as npt

from ts_semantic_feature_detector.features_2d.boxes import BoxGroup
from ts_semantic_feature_detector.features_2d.masks import Mask
from ts_semantic_feature_detector.features_2d.masks import MaskGroup

class DetectionGroup:
    """
    Agroup masks and stems groups to do filtering. 

    Attributes:
        boxes - a features_2d.boxes.BoxGroup object containing the boxes.
        masks - a features_2d.masks.MaskGroup object containing all the masks.
        scores - a Numpy array containing the inference scores.
    """

    def __init__(
        self,
        boxes: npt.ArrayLike,
        masks: npt.ArrayLike,
        scores: npt.ArrayLike,
        binary_threshold: float
    ):
        """
        Initializes a single detection.

        Args:
            boxes - a Numpy array containing the masks with shape
                (num_boxes, 4). This is the same format outputted 
                by the Mask RCNN network.
            masks - a Numpy array containing the masks with shape
                (num_masks, color_channel, height, width). This is the same
                format outputted by the Mask RCNN network.
            scores - a Numpy array containing the inference scores provided by
                the network. Masks and scores must be at the same corresponding
                order.
            binary_threshold - a float number representing the threshold with
                the masks will be binarized. Binary masks are extensively used
                in this project, so this argument is mandatory.
        """

        self.box_group = BoxGroup(boxes)
        self.mask_group = MaskGroup(masks, binary_threshold)
        self.scores = scores

    def is_empty(
        self
    ) -> bool:
        """
        Returns True if there is no detection in this group.
        """

        return len(self.mask_group.data) == 0

    def metric_filtering(
        self,
        type: str,
        score_threshold: float = 0.5,
        percentage: float = 0.5,
    ):
        """
        Filter the detection group by some metric.

        It discards the masks, boxes and scores considered worst.

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
                del self.mask_group.data[idx]
                del self.box_group.data[idx]
                idxs -= 1
        elif type == 'percentage':
            num_predictions = int(len(self.masks)*percentage)
            del self.mask_group.data[num_predictions:]
            del self.box_group.data[num_predictions:]
            np.delete(self.scores, range(num_predictions, len(self.scores)))

    def filter_redundancy(
        self,
        x_coordinate_threshold: float
    ):
        """
        Filter the detections that represents the same crop.

        The filtering follows these steps:
            1. Finds the appropriate curves to describe each mask. 
            For more information about how the curves are obtained,
            please refer to the features2d.Mask.extract_curves method 
            documentation.
            2. Finds the X coordinate where the obtained line intercepts
            the bottom part of the image.
            3. Applies a threshold on the distance between consecutive X
            coordinates. Masks that have this coordinate too close are
            merged and their curves are calculated again. 

        #TODO: Throw an exception when no detections are informed.

        Args:
            x_coordinate_threshold: a float containing the threshold to
                be applied to the distances between lines' X coordinates.
        """

        if self.scores.any():
            for mask in self.mask_group.data:
                # Garantees that the filtering is done only after the curves
                # are extracted.
                if mask.ransac_line is None:
                    mask.extract_curves()
                
                mask.x_bottom = mask.ransac_line.evaluate_line_at_y(mask.data.shape[0])

            # Sorts the masks and the corresponding boxes using the x_bottom property as the key.
            self.mask_group.data, self.box_group.data = zip(
                *sorted(
                    zip(
                        self.mask_group.data,
                        self.box_group.data
                    ),
                    key=lambda det: det[0].x_bottom 
                )
            )
            self.mask_group.data = list(self.mask_group.data)
            self.box_group.data = list(self.box_group.data)

            # Gets the x_bottom values and calculates the distance between them. Finds the indices
            # that are smaller than the threshold.
            xs_bottom = [mask.x_bottom for mask in self.mask_group.data]
            dist_between_x_bottom = np.abs(np.array(xs_bottom[0:-1]) - np.array(xs_bottom[1:]))
            dist_between_x_bottom_idx = (dist_between_x_bottom < x_coordinate_threshold).nonzero()[0]
            
            # Iterate over all redundant bottom_x points
            for idx in dist_between_x_bottom_idx:
                combined_mask_data = self.mask_group.data[idx].data + self.mask_group.data[idx + 1].data
                binary_threshold = self.mask_group.data[idx].binary_threshold
                
                # Converts the combined mask to the output inference shape.
                # Allows to use the default Mask constructor. 
                combined_mask_data = np.moveaxis(combined_mask_data, 2, 0)

                self.mask_group.data[idx] = Mask(combined_mask_data, binary_threshold)
                self.mask_group.data[idx].extract_curves()
                self.scores[idx] = np.average(self.scores[idx:idx+2])

                # Delete the redundant detection (mask and box).
                np.delete(self.scores, idx + 1)
                del self.mask_group.data[idx + 1]
                del self.box_group.data[idx + 1]
                dist_between_x_bottom_idx -= 1
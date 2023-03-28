"""
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy.typing as npt

class Box:
    """
    Contains a single box.

    Implements methods to process boxes and visualize data.

    Attributes:
        data - a Numpy array containing the box data with shape (1, 4)
    """

    def __init__(
        self,
        box: npt.ArrayLike
    ):
        """
        Initialize a single box.

        Args:
            box - a Numpy array containing the box data with shape (1, 4)
        """
        self.data = box

    def plot(
        self,
        ax: plt.Axes,
        color: str,
        linewidth = 3
    ):
        """
        Plot a single box using the Matplotlib library.

        Args:
            ax - a matplotlib.pyplot.Axes object containing the subplot where
                it's desired to insert the box.
            color - a string containing the desired hexadecimal color for the
                plot.
            linewidth - a float number indicating the box's line width.
        """

        ax.add_patch(
            patches.Rectangle(
                (
                    self.data[0],
                    self.data[1]
                ),
                self.data[2] - self.data[0],
                self.data[3] - self.data[1],
                edgecolor=color,
                linewidth=linewidth,
                facecolor = 'None'
            )
        )

class BoxGroup:
    """
    Agroup boxes to do filtering.

    The box filtering uses the resutls from the mask filtering.
    For more information, see ts_semantic_feature_detector.features_2d.masks
    
    Attributes:
        data - a list containing all the masks as features_2d.boxes.Box
            objects. 
        scores - a Numpy array containing the boxes' inference scores.
    """

    def __init__(
        self,
        boxes: npt.ArrayLike,
    ):
        """
        Initializes the group of boxes.

        Args:
            boxes - a Numpy array containing the masks with shape
                (num_boxes, 4). This is the same format outputted 
                by the Mask RCNN network.
        """

        self.data = []
        for box in boxes:
            self.data.append(Box(box))

    def plot(
        self,
        ax: plt.Axes,
        color: str,
        linewidth = 3
    ):
        """
        Plot the box group using the Matplotlib library.

        Args:
            ax - a matplotlib.pyplot.Axes object containing the subplot where
                it's desired to insert the box.
            color - a string containing the desired hexadecimal color for the
                plot.
            linewidth - a float number indicating the box's line width.
        """

        for box in self.data:
            box.plot(ax, color, linewidth)
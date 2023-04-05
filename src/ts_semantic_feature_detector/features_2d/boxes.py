"""
Implements basic operations for bounding boxes.

It is usefeul to visualize the results of the Mask RCNN network and to
have a class representation in the same way as the masks.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy.typing as npt

class Box:
    """
    Contains a single box.

    Implements methods to process boxes and visualize data.

    Attributes:
        data (:obj:`np.ndarray`): the box data with shape (1, 4)
    """

    def __init__(
        self,
        box: npt.ArrayLike
    ):
        """
        Initialize a single box.

        Args:
            box (:obj:`np.ndarray`): the box data with shape (1, 4)
        """
        self.data = box

    def plot(
        self,
        ax: plt.Axes,
        color: str,
        linewidth = 3.0
    ) -> None:
        """
        Plot a single box using the Matplotlib library.

        Args:
            ax (:obj:`matplotlib.pyplot.Axes`): the subplot where it's desired to 
                insert the box.
            color (str): the desired hexadecimal color for the plot.
            linewidth (float, optional): the box's line width.
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
        data (:obj:`list`): the masks as features_2d.boxes.Box objects. 
        scores (:obj:`np.ndarray`): the boxes' inference scores.
    """

    def __init__(
        self,
        boxes: npt.ArrayLike,
    ):
        """
        Initializes the group of boxes.

        Args:
            boxes (:obj:`np.ndarray`): the masks with shape (num_boxes, 4). 
                This is the same format outputted by the Mask RCNN network.
        """

        self.data = []
        for box in boxes:
            self.data.append(Box(box))

    def plot(
        self,
        ax: plt.Axes,
        color: str,
        linewidth: float = 3.0
    ) -> None:
        """
        Plot the box group using the Matplotlib library.

        Args:
            ax (:obj:`matplotlib.pyplot.Axes`): the subplot where it's desired to 
                insert the box.
            color (str): containing the desired hexadecimal color for the plot.
            linewidth (float, optional): the box's line width.
        """

        for box in self.data:
            box.plot(ax, color, linewidth)
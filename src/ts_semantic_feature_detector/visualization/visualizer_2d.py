"""
Implements 2D visualizations using Matplotlib library.
"""

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy.typing as npt
from PIL import Image

from ts_semantic_feature_detector.features_2d.masks import MaskGroup

class Visualizer2D:
    """
    Abstracts specific visualizations using Matplolib library.
    """

    def plot_mask_group(
        self,
        mask_group: MaskGroup,
        rgb_img: Union[Image.Image, npt.ArrayLike] = None,
        subplot_shape: Tuple[int, int] = None,
        opt_avg_curve: str = None,
        opt_ransac_line: str = None,
        alpha: float = 1.0,
    ) -> None:
        """
        Plot the group of masks.

        The masks can be showed separately or together.

        #TODO: check if the subplot_shape is compatible with the
            number of masks.

        Args:
            mask_group (:obj:`features2d.masks.MaskGroup`): the mask group.
            rgb_img (PIL.Image): the RGB image. If it is not informed, then 
                the masks are plotted with no RGB image in the back.
            subplot_shape (a tuple of [int, int], optional): mask distribution
                in the figure. If it is not informed, the masks are showed in the 
                same figure.
            opt_avg_curve (str, optional): the Matplotlib color and line options for 
                the average curve (see matplotlib.pyplot.plot documentation 
                for more details). If not specified, the curve will be omitted.
            opt_ransac_line (str, optional): the Matplotlib color and line options
                for the RANSAC line (see matplotlib.pyplot.plot documentation for more 
                details). If not specified, the line will be omitted.
            alpha (float, optional): the mask transparency amount.
        """
        plt.figure()

        if subplot_shape is None:
            if rgb_img is not None:
                plt.imshow(rgb_img)
            
            mask_group.plot()

            if opt_avg_curve is not None:
                for mask in mask_group.masks:
                    mask.average_curve.plot(options=opt_avg_curve)

            if opt_ransac_line is not None:
                for mask in mask_group.masks:
                    mask.ransac_line.plot(options=opt_ransac_line)
        else:
            num_masks = len(mask_group.masks)
            
            for i, mask in zip(range(num_masks), mask_group.masks):
                plt.subplot(subplot_shape[0], subplot_shape[1], i+1)
                
                if rgb_img is not None:
                    plt.imshow(rgb_img)

                mask.plot(alpha)

                if opt_avg_curve is not None:
                    mask.average_curve.plot(options=opt_avg_curve)
                if opt_ransac_line is not None:
                    mask.ransac_line.plot(options=opt_ransac_line)

    def show(
        self,
        show_maximized: bool = False
    ) -> None:
        """
        Shows the previously configured plots.

        Args:
            show_maximized (bool, optional): indicates if the plot
                will be showed with a maximized window.
        """
        if show_maximized:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.show()

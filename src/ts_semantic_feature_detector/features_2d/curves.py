"""
Implements basic operations for 2D curves.

The curves are used to filter redundant masks. 
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy.typing as npt

class Curve2D:
    """
    Abstraction of a 2D curve.

    Useful for storing xy points and to make visualization
    easier.

    Attributes:
        x (:obj:`np.ndarray`): the curve's x values.
        y (:obj:`np.ndarray`): the curve's y values.
    """
    
    def __init__(
        self,
        x: npt.ArrayLike = None,
        y: npt.ArrayLike = None
    ):
        """
        Initializes a curve.
        
        Args:
            x (:obj:`np.ndarray`, optional): the curve's x values.
            y (:obj:`np.ndarray`, optional): the curve's y values.
        """
        self.x = x
        self.y = y

    def plot(
        self,
        options: str = ''
    ) -> None:
        """
        Plot the curve using Matplolib library.

        Args:
            options (str, optional): the Matplotlib color and line options 
                (see :obj:`matplotlib.pyplot.plot` documentation for more 
                details).
        """
        plt.plot(self.x, self.y, options)

class Line2D(Curve2D):
    """
    Abstraction of a 2D line.

    It inherited from the Curve2D class. See documentation from 
    :obj:`Curve2D` for reference.

    Attributes:
        angular_coef (float): the line's angular coeficient.
        linear_coef: (float): the line's linear coeficient.
    """

    def __init__(
        self,
        angular_coef: float,
        linear_coef: float,
        x: npt.ArrayLike = None,
        y: npt.ArrayLike = None,
    ):
        """
        Initializes a line.

        Args:
            angular_coef (float): the line's angular coeficient.
            linear_coef (float): the line's linear coeficient.
            x (:obj:`np.ndarray`, optional): the curve's x values.
            y (:obj:`np.ndarray`, optional): the curve's y values.
        """
        super().__init__(x, y)
        self.angular_coef = angular_coef
        self.linear_coef = linear_coef

    def evaluate_line_at_x(
        self, 
        x: Union[float, npt.ArrayLike],
    ) -> Union[float, npt.ArrayLike]:
        """
        Evaluates the line at a specific x value

        Calculate the expression y = m*x + b, where m is the angular
        coeficient and b is the linear coeficient. 

        Args:
            x (float/:obj:`np.ndarray`): x coordinate(s) to evaluate the line.
        """
        return self.angular_coef*x + self.linear_coef

    def evaluate_line_at_y(
        self, 
        y: Union[float, npt.ArrayLike],
    ) -> Union[float, npt.ArrayLike]:
        """
        Evaluates the line at a specific y value

        Calculate the expression x = (y - b)/m, where m is the angular
        coeficient and b is the linear coeficient. 

        Args:
            y (float/:obj:`np.ndarray`): the y coordinate to evaluate the line.
        """
        return (y - self.linear_coef)/self.angular_coef

    def plot(self,
        x_values: npt.ArrayLike = None,
        y_values: npt.ArrayLike = None,
        options: str = ''
    ) -> None:
        """
        Plot the line using the Matplolib library.

        Args:
            x_values (:obj:`np.ndarray`, optional): the x coordinates to plot
                using the line coefficients. If x_values or y_values are
                not specified, the Curve2D.x and Curve2D.y attributes are 
                used to plot.
            y_values (:obj:`np.ndarray`, optional): the y coordinates to plot
                using the line coefficients. If x_values or y_values are
                not specified, the Curve2D.x and Curve2D.y attributes are 
                used to plot.
            options (str, optional): the Matplotlib color and line options 
                (see :obj:`matplotlib.pyplot.plot` documentation for more 
                details).
        """
        if x_values is not None:
            plt.plot(x_values, self.evaluate_line_at_x(x_values), options)
        elif y_values is not None:
            plt.plot(self.evaluate_line_at_y(y_values), y_values, options)
        elif self.x is None or self.y is None:
            if self.x is None:
                plt.plot(self.evaluate_line_at_y(self.y), self.y, options)
            elif self.y is None:
                plt.plot(self.x, self.evaluate_line_at_x(self.x), options)
        else:
            plt.plot(self.x, self.y, options)
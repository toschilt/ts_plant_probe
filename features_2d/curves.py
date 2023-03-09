"""
"""
import numpy.typing as npt

class Curve2D:
    """
    Abstraction of a 2D curve.

    Useful for storing xy points and to make visualization
    easier.

    Attributes:
        x: a Numpy array with the curve's x values.
        y: a Numpy array with the curve's y values.
    """
    
    def __init__(
        self,
        x: npt.ArrayLike = None,
        y: npt.ArrayLike = None
    ) -> None:
        """
        Initializes a curve.
        
        Args:
            x: a Numpy array with the curve's x values.
            y: a Numpy array with the curve's y values.
        """
        self.x = x
        self.y = y

class Line2D(Curve2D):
    """
    Abstraction of a 2D line.

    It inherited from the Curve2D class. See documentation for
    reference.

    Attributes:
        angular_coef: a float number containing the line's angular
            coeficient.
        linear_coef: a flot number containing the line's linear
            coeficient.
    """

    def __init__(
        self,
        angular_coef: float,
        linear_coef: float,
        x: npt.ArrayLike = None,
        y: npt.ArrayLike = None,
    ) -> None:
        """
        Initializes a line.

        Args:
            x: a Numpy array with the curve's x values.
            y: a Numpy array with the curve's y values.
            angular_coef: a float number containing the line's angular
                coeficient.
            linear_coef: a flot number containing the line's linear
                coeficient.
        """
        super().__init__(x, y)
        self.angular_coef = angular_coef
        self.linear_coef = linear_coef

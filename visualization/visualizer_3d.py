"""
"""

import typing as List

import plotly.graph_objects as go

class Visualizer3D:
    """
    Abstract specific visualizations using Plotly library.
    """
    
    def __init__(
        self,
        data: List = None
    ) -> None:
        """
        Initializes the Visualizer3D.

        Args:
            data: a list containing existing plots.
        """

        self.data = []
        if data is not None:
            self.data = data

    def show(self):
        """
        Shows the previously configured plots.
        """

        fig = go.Figure(data=self.data)
        fig.show()
"""
"""

import io
import numpy as np
import matplotlib.pyplot as plt

class TimerVisualizer:
    """
    Visualizes the times of each step of the pipeline
    """

    def __init__(
        self,
        times_file: str,
    ):
        """
        Initializes the visualizer

        Args:
            times_file: a string containing the path to the times file
        """

        self.times_file = times_file

        with open(self.times_file, 'r') as f:
            self.headers = f.readline()
            self.headers = self.headers.split(',')

            self.data = np.loadtxt(
                self.times_file,
                delimiter=',',
                skiprows=1,
                usecols=range(0, len(self.headers))
            )

    def plot(
        self
    ):
        """
        Plots the times of each step of the pipeline
        """

        plt.figure()
        for header, data in zip(self.headers, self.data.T):
            if (data > 0.001).all():
                plt.plot(data, label=header)
        plt.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    visualizer = TimerVisualizer(
        '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/output/times.txt'
    )
    visualizer.plot()
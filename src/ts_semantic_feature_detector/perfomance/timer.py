"""
Implementation of a timer class to measure the performance of the
pipeline.
"""

import time

class Timer:
    """
    Timer class to measure each step of the pipeline

    Attributes:
        measurements (a dict [str, List]): the measurements of each
            timer. The key is the name of the timer and the value is a
            list containing the measurements.
        start_times: (a dict [str, List]): the start time of each
            timer. The key is the name of the timer and the value is a
            list containing the start time.
        end_times: (a dict [str, List]): the end time of each
            timer. The key is the name of the timer and the value is a
            list containing the end time.
    """

    def __init__(
        self
    ):
        """
        Initializes the timer
        """

        self.measurements = {}
        
        self.start_times = {}
        self.end_times = {}

    def new_cicle(
        self
    ):
        """
        Resets the start and end times
        """
        self.measurements.clear()
        self.start_times.clear()
        self.end_times.clear()

    def start(
        self,
        name: str
    ):
        """
        Starts a timer with a given name

        Args:
            name (str): the name of the timer.
        """
        self.start_times[name] = time.time()

    def stop(
        self,
        name: str
    ):
        """
        Stops a timer with a given name and saves the measurement.

        Args:
            name (str): the name of the timer.
        """

        self.end_times[name] = time.time()

        if name not in self.measurements.keys():
            self.measurements[name] = []

        self.measurements[name].append(
            self.end_times[name] - self.start_times[name]
        )
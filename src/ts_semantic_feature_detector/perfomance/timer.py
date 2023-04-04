"""
"""

import time

class Timer:
    """
    Timer class to measure each step of the pipeline

    Attributes:
        measurements: a list of dictionaries containing the name of
            the timer, the start time, the end time and the total
            time of each timer.
        start_times: a dictionary containing the start time of each
            timer.
        end_times: a dictionary containing the end time of each
            timer.
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

    def start(
        self,
        name: str
    ):
        """
        Starts a timer with a given name

        Args:
            name: a string containing the name of the timer
        """

        self.start_times[name] = time.time()

    def stop(
        self,
        name: str
    ):
        """
        Stops a timer with a given name and saves the measurement.

        Args:
            name: a string containing the name of the timer
        """

        self.end_times[name] = time.time()

        if name not in self.measurements.keys():
            self.measurements[name] = []

        self.measurements[name].append(
            self.end_times[name] - self.start_times[name]
        )
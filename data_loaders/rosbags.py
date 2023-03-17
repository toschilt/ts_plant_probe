"""
"""

from operator import itemgetter
from typing import Dict
import yaml

import rosbag
import rospy

class AgricultureRosbagLoader:
    """
    Implements some useful functions to load rosbags.

    Attributes:
        data: a rosbag.Bag object containing the bag abstraction.
        topics: a dictionary containing the names of the topics desired to
            sync.
        lowest_freq_topic: a string containing the name of the lowest frequent
            topic.
        start_time: a float containing the start time of the recorded
            data.
        
    """

    def __init__(
        self,
        bag_path: str,
        topics: Dict,
        lowest_freq_topic: str,
        skip_seconds: float = None
    ):
        """
        Initializes the rosbag loader.

        Args:
            bag_path: the path to the rosbag filepath.
            topics: a dictionary containing the names of the topics desired to
                sync. The same keys will be used to return data to the user.
            lowest_freq_topic: a string containing the name of the lowest frequent
                topic. It will be used as a landmark to sync bits of data.
            skip_seconds: a float containing the amount of seconds that
                is desired to skip from the bag's beginning. If it is ommited,
                the bag is played entirely.

        #TODO: Find the lowest frequency topic automatically.
        """

        self.bag = rosbag.Bag(bag_path)
        self.topics = topics
        self.lowest_freq_topic = lowest_freq_topic

        self.start_time = self.bag.get_start_time()
        if skip_seconds is not None:
            self.start_time += skip_seconds
        self.start_time = rospy.Time.from_sec(self.start_time)

        self.raw_data = self.bag.read_messages(
            topics=self.topics,
            start_time=self.start_time
        )
        
    def get_sync_data(self):
        """
        A generator that yields the topics syncronized.
        """
        msgs = {}
        for key in self.topics.keys():
            msgs[key] = []
            
        for topic, msg, t in self.raw_data:
            # Storage every message inside respective field in dictionary.
            for key in msgs.keys():
                if topic == self.topics[key]:
                    msgs[key].append(msg)

            # Use the most low frequency topic to synchronize the topics.
            if topic == self.lowest_freq_topic:
                # Checks if all the topics has at least one single message.
                # If one of them do not, we need to discard this batch.
                batch_invalid = False
                for key in msgs.keys():
                    if not msgs[key]:
                        batch_invalid = True

                if not batch_invalid:
                    filtered_topics = {}

                    # For each topic, find the message that is closer to the
                    # current time.
                    for key in msgs.keys():
                        diff_times = [abs(t - msg_.header.stamp) for msg_ in msgs[key]]
                        closer_time_idx = min(enumerate(diff_times), key=itemgetter(1))[0]
                        filtered_topics[key] = msgs[key][closer_time_idx]

                    # Clear the temporary messages to avoid memory leaking
                    for key in self.topics.keys():
                        msgs[key].clear()

                    yield filtered_topics
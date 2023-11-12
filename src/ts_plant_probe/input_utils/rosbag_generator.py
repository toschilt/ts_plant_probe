from typing import Generator, List

import rosbag

class RosbagGenerator():
    """
    Creates a customized rosbag file reader.    
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.bag = rosbag.Bag(filename)

    def read(self, topics: List[str]) -> Generator:
        """
        Reads the rosbag file according to the given topics.

        Args:
            topics (List[str]): the topics to be read.

        Yields:
            Dict[str, Any, rospy.Time]: a dictionary containing the
                topic, the message and the timestamp.

        """
        for topic, msg, t in self.bag.read_messages(topics=topics):
            payload = {}
            payload['topic'] = topic
            payload['msg'] = msg
            payload['t'] = t
            yield payload

    def read_buffer(
        self,
        topics: List[str],
        buffer_size: int
    ) -> Generator:
        """
        Reads the rosbag file according to the given topics while
        buffering the messages.

        Args:
            topics (List[str]): the topics to be read.
            buffer_size (int): the buffer size.

        Yields:
            Vector[Dict[str, Any, rospy.Time]]: a vector of dictionaries
                containing the topic, the message and the timestamp.

        """
        i = 0
        buffer = []

        for payload in self.read(topics=topics):
            buffer.append(payload)
            i += 1

            if i == buffer_size:
                yield buffer
                buffer = []
                i = 0
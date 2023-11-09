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
            Tuple[str, Any, rospy.Time]: the topic name, the message and the
                timestamp.

        """
        for topic, msg, t in self.bag.read_messages(topics=topics):
            yield topic, msg, t

if __name__ == '__main__':
    rosbag_generator = RosbagGenerator(
        '/home/ltoschi/Documents/LabRoM/plant_by_plant_detection/ts_2023_08_04_12h58m41s_two_rows.bag'
    )

    for topic, msg, t in rosbag_generator.read(['/tf']):
        print(topic, msg, t)
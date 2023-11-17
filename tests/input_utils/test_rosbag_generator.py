import pytest

import rosbag
from std_msgs.msg import Int32

from ts_plant_probe.input_utils.rosbag_generator import RosbagGenerator

@pytest.fixture(scope="module")
def rosbag_path():
    return './tests/input_utils/resources/test.bag'

@pytest.fixture(scope="module", autouse=True)
def example_bag(rosbag_path):
    ''' Creates an example rosbag file. '''
    bag = rosbag.Bag(rosbag_path, 'w')

    for i in range(10):
        msg = Int32()
        msg.data = i
        bag.write('chatter', msg)

    bag.close()

class TestRosbagGenerator():
    @pytest.fixture(scope="class")
    def rosbag_generator(self, rosbag_path):
        return RosbagGenerator(rosbag_path)
        
    def test_bag_read(
        self,
        rosbag_generator
    ):
        payload = next(rosbag_generator.read(['chatter']))
        assert payload['topic'] == 'chatter'
        assert payload['msg'].data == 0
        assert payload['t'].secs > 0

    def test_bag_read_buffer(
        self,
        rosbag_generator
    ):
        buffer = next(rosbag_generator.read_buffer(['chatter'], 10))
        assert len(buffer) == 10
        assert buffer[0]['msg'].data == 0
        assert buffer[9]['msg'].data == 9
        assert buffer[0]['t'].secs > 0
        assert buffer[9]['t'].secs > 0
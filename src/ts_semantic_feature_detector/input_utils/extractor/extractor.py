"""
"""
import os

from typing import Dict, List, Tuple

from cv_bridge import CvBridge
import cv2

import rosbag
import roslaunch
import rospkg
import rospy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

class DataExtractor():
    """
    Implements a customized rosbags and SVO data extractor.

    This extractor is not intended for general use. It has
    pre-defined topics needed by this project.
    """

    def __init__(
        self,
        data_path: str,
        rosbag_file: str,
        svo_file: str
    ):
        """
        Initializes the data extractor.

        Args:
            data_path: a string containing the path to the the upper folder 
                where the compressed and the extracted data are storaged. It
                is expected a folder arrangement similar to:

                /data
                    /rosbag
                        - .bag file
                    /svo
                        - .svo file
            rosbag_file: a string containing the .bag file name.
            svo_file: a string containing the .svo file name.
        """
        
        rosbag_topics, svo_topics = self._get_topics_names()
        
        # Rosbag extractor
        rospy.loginfo('Starting to extract the rosbag...')
        bag = rosbag.Bag(data_path + '/rosbag/' + rosbag_file)
        self._write_rosbag_raw_data(bag, rosbag_topics, data_path)

        # SVO extractor
        rospy.loginfo('Starting to extract the SVO file...')
        self.bridge = CvBridge()
        svo_subscribers = self._get_svo_subscribers(svo_topics, data_path)
        launch = self._write_svo_raw_data(svo_file, data_path)

    def _get_topics_names(
        self
    ) -> Tuple[Dict, Dict]:
        """
        Defines the topics names.

        It is intended to be changed manually if needed. This class code
        needs to be changed to add/remove ROS subscribers and callbacks.

        Returns:
            a pair of dictionaries. The first one describes the topics
            to be extracted from the rosbag and the second one describes
            the topics to be extracted from the SVO file.
        """
        rosbag_topics = {
            'ekf': '/terrasentia/ekf',
            # 'imu': '/terrasentia/imu',
            'imu': '/terrasentia/zed2/zed_node/imu/data',
            # 'rgb': '/terrasentia/zed2/zed_node/left/image_rect_color/compressed',
            # 'depth': '/terrasentia/zed2/zed_node/depth/depth_registered'
        }

        svo_topics = {
            # 'pose': '/zed2/zed_node/pose',
            'rgb': '/zed2/zed_node/rgb/image_rect_color',
            'depth': '/zed2/zed_node/depth/depth_registered'
        }

        return rosbag_topics, svo_topics

    def _write_rosbag_raw_data(
        self,
        bag: rosbag.Bag,
        topics: Dict,
        data_path: str
    ):
        """
        Writes all desired data to the data folder.

        Args:
            bag: a rosbag.Bag object containing the data.
            topics: a dictionary containing all the desired topics
                published by the rosbag.
            data_path: a string containing the path to the the upper folder 
                where the compressed and the extracted data are storaged.
        """
        topics_list = []
        for topic in topics.values():
            topics_list.append(topic)
        raw_data = bag.read_messages(topics=topics_list)

        ekf_path = data_path + '/rosbag/ekf/'
        self._create_folder_if_needed(ekf_path)
        ekf_f = open(ekf_path + 'ekf.csv', 'w+')
        ekf_f.write(
            'secs,nsecs,' +
            'position.x,position.y,position.z,' +
            'orientation.x,orientation.y,orientation.z,orientation.w' +
            '\n'
        )

        imu_path = data_path + '/rosbag/imu/'
        self._create_folder_if_needed(imu_path)
        imu_f = open(imu_path + 'imu.csv', 'w+')
        imu_f.write(
            'secs,nsecs,' +
            'orientation.x,orientation.y,orientation.z,orientation.w' +
            '\n'
        )

        for topic, msg, __ in raw_data:
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs

            if topic == topics['ekf']:
                position = msg.pose.pose.position
                position = [position.x, position.y, position.z]
                orientation = msg.pose.pose.orientation
                orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
                
                ekf_f.write(str(secs))
                ekf_f.write(',')
                ekf_f.write(str(nsecs))
                ekf_f.write(',')
                for pos in position:
                    ekf_f.write(str(pos))
                    ekf_f.write(',')
                for ori in orientation[:-1]:
                    ekf_f.write(str(ori))
                    ekf_f.write(',')
                ekf_f.write(str(orientation[-1]))
                ekf_f.write('\n')

            if topic == topics['imu']:
                orientation = msg.orientation
                orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
                
                imu_f.write(str(secs))
                imu_f.write(',')
                imu_f.write(str(nsecs))
                imu_f.write(',')
                for ori in orientation[:-1]:
                    imu_f.write(str(ori))
                    imu_f.write(',')
                imu_f.write(str(orientation[-1]))
                imu_f.write('\n')

        ekf_f.close()
        imu_f.close()
                
    def _create_folder_if_needed(
        self,
        path: str
    ):
        """
        Creates a folder if it does not exist yet.

        Args:
            path: a string containing the path to the desired folder.

        #FIXME: permission problem when creating folders.
        """
        pass
        # if not os.path.exists(path):
        #     os.umask(0)
        #     os.makedirs(path)

    def _get_svo_subscribers(
        self,
        topics: Dict,
        data_path: str
    ) -> List:
        """
        Defines the ROS subscribers to get SVO data. 

        It is intended to be changed manually if needed.

        Args:
            topics: a dictionary containing all the desired topics
                published by the SVO.
            data_path: a string containing the path to the the upper folder 
                where the compressed and the extracted data are storaged.
                
        Returns:
            a list containing all the ROS subscribers.
        """

        svo_subscribers = []
        # svo_subscribers.append(
        #     rospy.Subscriber(
        #         self.svo_topics['pose'],
        #         PoseStamped,
        #         self._svo_pose_cb,
        #         queue_size=5
        #     )
        # )

        rgb_path = data_path + '/svo/rgb/'  
        self._create_folder_if_needed(rgb_path)
        svo_subscribers.append(
            rospy.Subscriber(
                topics['rgb'],
                Image,
                self._svo_img_cb,
                rgb_path,
                queue_size=5
            )
        )

        depth_path = data_path + '/svo/depth/'
        self._create_folder_if_needed(depth_path)
        svo_subscribers.append(
            rospy.Subscriber(
                topics['depth'],
                Image,
                self._svo_img_cb,
                depth_path,
                queue_size=5
            )
        )

        return svo_subscribers

    def _svo_img_cb(self, msg, path):
        """
        Callback for SVO image messages. 

        It saves the images as PNG files with corresponding time stamps.
        """
        secs = msg.header.stamp.secs
        nsecs = msg.header.stamp.nsecs

        file_name = path + str(secs) + '_n' + str(nsecs).zfill(10) + '.png'
        img = self.bridge.imgmsg_to_cv2(msg)
        cv2.imwrite(file_name, img)

    def _write_svo_raw_data(
        self,
        svo_file: str,
        data_path: str 
    ) -> roslaunch.parent.ROSLaunchParent:
        """
        Writes all desired data to the data folder.

        Args:
            svo_file: a string containing the name of the SVO file.
            data_path: a string containing the path to the the upper folder 
                where the compressed and the extracted data are storaged.

        Returns:
            the roslaunch.parent.ROSLaunchParent object. It can be
            used to stop the nodes from the executable.
        """
        rospack = rospkg.RosPack()
        zed2_launch_path = rospack.get_path('zed_wrapper') + '/launch/zed2.launch'

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        print('svo_file:=' + data_path + '/svo/' + svo_file)
        launch = roslaunch.parent.ROSLaunchParent(
            uuid,
            [
                (
                    zed2_launch_path, 
                    ['svo_file:=' + data_path + '/svo/' + svo_file]
                )
            ]
        )
        launch.start()

        return launch
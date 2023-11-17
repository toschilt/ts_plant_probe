import rospy

from ts_semantic_feature_detector.input_utils.extractor.extractor import DataExtractor

def main():
    rospy.init_node('extractor')
    
    DataExtractor(
        '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/data',
        'ts_2022_09_01_11h44m59s_one_row.bag',
        'ts_2022_09_01_11h44m59s_one_row.svo'
    )

    rospy.spin()

if __name__ == '__main__':
    main()

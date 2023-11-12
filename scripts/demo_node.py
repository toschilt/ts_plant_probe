import rospy

from ts_plant_probe.input_utils.rosbag_generator import RosbagGenerator

if __name__ == '__main__':
    rospy.init_node('demo_node')
    file = '/home/ltoschi/Documents/LabRoM/plant_by_plant_detection/ts_2023_08_04_12h58m41s_two_rows.bag'
    rg = RosbagGenerator(file)

    for payload in rg.read(['/tf']):
        print(payload)
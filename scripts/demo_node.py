import logging

import rospy

from ts_plant_probe.input_utils.rosbag_generator import RosbagGenerator
from ts_plant_probe.input_utils.parameter_parse import ParameterParse
from ts_plant_probe.seg_model.dataset import TerraSentiaDataset

if __name__ == '__main__':
    rospy.init_node('demo_node')
    
    # Create a parameter parser
    params = ParameterParse().parameters

    logging.config.fileConfig(params['logging_conf_path'])

    tsdata = TerraSentiaDataset(
        png_path=params['dataset_path'] + '/PNGImages',
        mask_path=params['dataset_path'] + '/Masks',
        metrics_path=params['dataset_metrics_path']
    )

    print('oi')

    rospy.spin()
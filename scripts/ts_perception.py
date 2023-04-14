import signal
import sys

from cv_bridge import CvBridge
import numpy as np
import rospy
import rospkg

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

from ts_semantic_feature_detector.features_2d.detections import DetectionGroup
from ts_semantic_feature_detector.features_3d.camera import StereoCamera
from ts_semantic_feature_detector.features_3d.crop import CornCropGroup
from ts_semantic_feature_detector.features_3d.ground_plane import GroundPlane
from ts_semantic_feature_detector.features_3d.scene import AgriculturalScene
from ts_semantic_feature_detector.features_3d.sequence import AgriculturalSequence
from ts_semantic_feature_detector.input_utils.extrinsics import get_extrinsics
from ts_semantic_feature_detector.input_utils.loaders.sync_loader import SynchronizedLoader
from ts_semantic_feature_detector.segmentation_model.model.mask_rcnn_stem_segmentation import MaskRCNNStemSegmentationModel
from ts_semantic_feature_detector.segmentation_model.ts_dataset.ts_load_dataset import TerraSentiaDataset
from ts_semantic_feature_detector.features_2d.tracking import *
from ts_semantic_feature_detector.visualization.colors import get_color_from_cluster
from ts_semantic_feature_detector.visualization.visualizer_2d import Visualizer2D
from ts_semantic_feature_detector.visualization.visualizer_3d import Visualizer3D
from ts_semantic_feature_detector.perfomance.timer import Timer
from ts_semantic_feature_detector.output_utils.writer import OutputWriter

class TerraSentiaPerception:

    def __init__(
        self
    ):
        rospy.init_node('inference')
        self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()
        self.sequence = AgriculturalSequence()
        self.v_3d = Visualizer3D()

        signal.signal(signal.SIGINT, self.signal_handler)
        
        rospy.loginfo('Getting data...')
        self.sync_loader = SynchronizedLoader(
            '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/data - cornfield1 - 09 01 - start 1000'
        )

        rospy.loginfo('Loading segmentation model aspects...')
        self.dataset = TerraSentiaDataset(
            self.rospack.get_path('ts_semantic_feature_detector') + '/src/ts_semantic_feature_detector/segmentation_model/ts_dataset/data/PNGImages',
            self.rospack.get_path('ts_semantic_feature_detector') + '/src/ts_semantic_feature_detector/segmentation_model/ts_dataset/data/StemPlantMasks',
            mean=[0.3470, 0.4711, 0.3395],
            std_dev=[0.2194, 0.2355, 0.2541],
        )

        rospy.loginfo('Loading segmentation model...')
        self.model = MaskRCNNStemSegmentationModel(
            self.dataset,
            450,
            800,
            model_path=self.rospack.get_path('ts_semantic_feature_detector') + '/src/ts_semantic_feature_detector/segmentation_model/model/checkpoints/model_better_mAP_122'
        )

        rospy.loginfo('Loading camera...')
        fx = 527.0302734375
        fy = 527.0302734375
        cx = 627.5240478515625
        cy = 341.2162170410156
        width = 1280
        height = 720
        self.camera = StereoCamera([fx, fy, cx, cy], [width, height])

        rospy.loginfo('Loading tracker...')
        self.tracker = AgricultureSort(\
            self.camera,
            max_age=2,
            min_hits=0,
            iou_threshold=0.1
        )

        rospy.loginfo('Loading timer...')
        self.timer = Timer()

        rospy.loginfo('Loading output writer...')
        self.output_writer = OutputWriter(
            self.rospack.get_path('ts_semantic_feature_detector') + '/output/odometry_factors.txt',
            self.rospack.get_path('ts_semantic_feature_detector') + '/output/emerging_points.txt',
            self.rospack.get_path('ts_semantic_feature_detector') + '/output/times.txt',
            ','
        )

        rospy.loginfo('Getting started...')
        self.main()      
    
    def main(self):
        see_sequence = 15
        for data in self.sync_loader.get_sync_data(1000):
            rospy.loginfo(f'Getting agricultural scene [{data["index"]}]...')

            rospy.loginfo('Getting extrinsics...')
            self.timer.start('get_extrinsics')
            p_world_body, orient_world_body, p_camera_body, orient_camera_body = get_extrinsics(
                data['ekf'],
                data['imu']
            )
            self.timer.stop('get_extrinsics')

            rospy.loginfo('Writing robot pose...')
            self.timer.start('write_odometry_factors')
            self.output_writer.write_odometry_factors(
                data['index'],
                data['ekf']
            )
            self.timer.stop('write_odometry_factors')
            
            rospy.loginfo('Getting masks and boxes...')
            self.timer.start('inference')
            __, boxes, masks, scores = self.model.inference(data['rgb'])
            self.timer.stop('inference')

            self.timer.start('detections')
            detections = DetectionGroup(
                boxes,
                masks,
                scores,
                binary_threshold=0.5
            )
            detections.metric_filtering('score', score_threshold=0.5)
            detections.filter_redundancy(x_coordinate_threshold=20)
            self.timer.stop('detections')

            # Check if there are any valid detections.
            if not detections.is_empty():
                rospy.loginfo('Getting the ground plane...')
                self.timer.start('ground_plane')
                gp = GroundPlane(
                    data['rgb'],
                    'threshold_gaussian',
                    self.camera,
                    data['depth'],
                    {
                        'hLow': 74,
                        'sLow': 0,
                        'vLow': 3,
                        'sHigh': 63,
                        'hHigh': 205,
                        'vHigh': 203,
                        'gaussian_filter': 12
                    }
                )
                self.timer.stop('ground_plane')

                rospy.loginfo('Getting the 3D points...')
                self.timer.start('corn_crop_group')
                crop_group = CornCropGroup(
                    detections,
                    self.camera,
                    data['depth'],
                    depth_neighbors=2,
                    ground_plane=gp
                )
                self.timer.stop('corn_crop_group')

                rospy.loginfo('Getting the agricultural scene...')
                self.timer.start('scene')
                scene = AgriculturalScene(data['index'], crop_group, gp)
                self.timer.stop('scene')

                rospy.loginfo('Downsampling point clouds...')
                self.timer.start('downsample_scene')
                scene.downsample(
                    crop_voxel_size=0.05,
                    ground_plane_voxel_size=0.01
                )
                self.timer.stop('downsample_scene')

                rospy.loginfo('Adding extrinsics to the 3D points...')
                self.timer.start('add_extrinsics_information')
                scene.add_extrinsics_information(
                    p_world_body,
                    orient_world_body,
                    p_camera_body,
                    orient_camera_body
                )
                self.timer.stop('add_extrinsics_information')

                self.timer.start('add_scene')
                self.sequence.add_scene(scene)
                self.timer.stop('add_scene')

                # rospy.loginfo('Tracking boxes...')
                # self.tracker.step(sequence)

                rospy.loginfo('Clustering crops...')
                self.timer.start('cluster_crops')
                self.sequence.cluster_crops()
                self.timer.stop('cluster_crops')

                rospy.loginfo('Filtering old clusters...')
                self.timer.start('remove_old_clusters')
                old_cluster_exist = self.sequence.remove_old_clusters(cluster_max_age=5)
                self.timer.stop('remove_old_clusters')

                rospy.loginfo('Finding old scenes...')
                self.timer.start('get_old_scenes_idxs')
                old_scenes_idxs = self.sequence.get_old_scenes_idxs(old_cluster_exist)
                self.timer.stop('get_old_scenes_idxs')

                rospy.loginfo('Writing emerging points from old scenes...')
                self.timer.start('write_emerging_points')
                self.output_writer.write_emerging_points(
                    self.sequence,
                    old_scenes_idxs
                )
                self.timer.stop('write_emerging_points')

                rospy.loginfo('Removing old scenes...')
                self.timer.start('remove_old_scenes')
                self.sequence.remove_old_scenes(old_scenes_idxs)
                self.timer.stop('remove_old_scenes')

                rospy.loginfo('Writing times...')
                self.output_writer.write_times(self.timer)

            # if self.sequence.scenes:
            #     rospy.loginfo('Plotting...')

            #     self.plot_3d(
            #         data,
            #         self.sequence,
            #         self.v_3d,
            #         see_sequence=see_sequence
            #     )

        self.output_writer.finish(self.sequence)

    def plot_3d(
        self,
        data,
        sequence,
        v_3d,
        see_sequence
    ):

        if data['index'] % see_sequence == 0:
            v_3d.data.clear()

            sequence.plot(
                data_plot=v_3d.data,
                line_scalars=np.linspace(-0.5, 0.5, 100),
                # plane_scalars=(
                #     np.linspace(-500, 600, 100),
                #     np.linspace(0, 1300, 100)
                # ),
                plot_3d_points_crop=False,
                plot_emerging_points=True,
                plot_3d_points_plane=False
            )
                
            v_3d.show()

    def signal_handler(self, sig, frame):
        rospy.loginfo('Please wait to finish writing the output files...')
        self.output_writer.finish(self.sequence)

        rospy.loginfo('Exiting...')
        sys.exit(0)

if __name__ == '__main__':
    TerraSentiaPerception()
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
from ts_semantic_feature_detector.output_utils.writer import OutputWriter

class TerraSentiaPerception:

    def __init__(
        self
    ):
        rospy.init_node('inference')
        self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()
        signal.signal(signal.SIGINT, self.signal_handler)
        
        rospy.loginfo('Getting data...')
        self.sync_loader = SynchronizedLoader(
            '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/data'
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

        rospy.loginfo('Loading output writer...')
        self.output_writer = OutputWriter(
            self.rospack.get_path('ts_semantic_feature_detector') + '/output/odometry_factors.txt',
            self.rospack.get_path('ts_semantic_feature_detector') + '/output/emerging_points.txt',
            ','
        )

        rospy.loginfo('Getting started...')
        self.main()      
    
    def main(self):
        sequence = AgriculturalSequence()
        v_3d = Visualizer3D()

        see_sequence = 10
        for data in self.sync_loader.get_sync_data():
            rospy.loginfo(f'Getting agricultural scene [{data["index"]}]...')

            rospy.loginfo('Getting extrinsics...')
            p_world_body, orient_world_body, p_camera_body, orient_camera_body = get_extrinsics(
                data['ekf'],
                data['imu']
            )

            rospy.loginfo('Writing robot pose...')
            self.output_writer.write_odometry_factors(
                data['index'],
                data['ekf']
            )
            
            rospy.loginfo('Getting masks and boxes...')
            __, boxes, masks, scores = self.model.inference(data['rgb'])

            detections = DetectionGroup(
                boxes,
                masks,
                scores,
                binary_threshold=0.5
            )
            detections.metric_filtering('score', score_threshold=0.5)
            detections.filter_redundancy(x_coordinate_threshold=20)

            # Check if there are any valid detections.
            if not detections.mask_group.data:
                continue

            rospy.loginfo('Getting the ground plane...')
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

            rospy.loginfo('Getting the 3D points...')
            crop_group = CornCropGroup(
                detections,
                self.camera,
                data['depth'],
                mask_filter_threshold=2,
                ground_plane=gp
            )

            rospy.loginfo('Adding extrinsics to the 3D points...')
            scene = AgriculturalScene(crop_group, gp)
            scene.add_extrinsics_information(
                p_world_body,
                orient_world_body,
                p_camera_body,
                orient_camera_body
            )
            sequence.add_scene(scene)

            # rospy.loginfo('Tracking boxes...')
            # self.tracker.step(sequence)

            rospy.loginfo('Clustering crops...')
            sequence.cluster_crops()

            rospy.loginfo('Writing emerging points...')
            self.output_writer.write_emerging_points(
                data['index'],
                scene
            )

            rospy.loginfo('Filtering old scenes...')
            sequence.remove_old_scenes(max_age=200)

            rospy.loginfo('Plotting...')
            # self.plot_tracking(
            #     data,
            #     sequence,
            #     detections,
            #     plot_predictions=True,
            #     save_fig=False
            # )
            self.plot_3d(
                data,
                sequence,
                v_3d,
                see_sequence=see_sequence
            )

    def plot_tracking(
        self, 
        data,
        sequence,
        detections,
        plot_predictions=False,
        save_fig=False,
    ):
        fig, ax = plt.subplots()
        ax.imshow(data['rgb'])
        detections.mask_group.plot(0.5)

        # Plot box detections
        for crop in sequence.scenes[-1].crop_group.crops:
            track = crop.crop_box.data
            ax.add_patch(
                patches.Rectangle(
                (track[0], track[1]),
                track[2] - track[0],
                track[3] - track[1],
                linewidth=3,
                edgecolor=get_color_from_cluster(int(crop.cluster)),
                facecolor='none')
            )

        # Plot predicted boxes
        if plot_predictions and len(sequence.scenes) > 1:
            for crop in sequence.scenes[-2].crop_group.crops:
                track = crop.crop_box.data
                offset = crop.estimated_motion_2d
                ax.add_patch(
                    patches.Rectangle(
                    (track[0] + offset[0], track[1] + offset[1]),
                    track[2] + offset[0] - track[0],
                    track[3] + offset[1] - track[1],
                    linewidth=3,
                    edgecolor='#FF0000',
                    facecolor='none')
                )
        
        if save_fig:
            plt.savefig(
                '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/log/'
                + str(data['index'])
                + '.png'
            )
        else:
            plt.show()

        plt.close()

    def plot_3d(
        self,
        data,
        sequence,
        v_3d,
        see_sequence
    ):

        if data['index'] % see_sequence == 0:
            rospy.loginfo('Plotting sequence...')
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
                plot_3d_points_plane=False,
                cluster_threshold=3
            )
                
            v_3d.show()

    def signal_handler(self, sig, frame):
        print('Exiting...')
        sys.exit(0)

if __name__ == '__main__':
    TerraSentiaPerception()
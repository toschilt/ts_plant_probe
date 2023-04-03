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
from ts_semantic_feature_detector.visualization.visualizer_2d import Visualizer2D
from ts_semantic_feature_detector.visualization.visualizer_3d import Visualizer3D

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

        rospy.loginfo('Getting started...')
        self.main()      
    
    def main(self):
        sequence = AgriculturalSequence()
        v_3d = Visualizer3D()

        see_sequence = 10
        for data in self.sync_loader.get_sync_data(1000):
            rospy.loginfo('Getting agricultural scene...')

            rospy.loginfo('Getting masks and boxes...')
            __, boxes, masks, scores = self.model.inference(data['rgb'])

            # Check if at least one mask was detected. If not, skip the frame.
            if not masks.any():
                continue

            detections = DetectionGroup(
                boxes,
                masks,
                scores,
                binary_threshold=0.5
            )
            detections.metric_filtering('score', score_threshold=0.5)
            detections.filter_redundancy(x_coordinate_threshold=20)

            rospy.loginfo('Getting extrinsics...')
            p_world_body, orient_world_body, p_camera_body, orient_camera_body = get_extrinsics(
                data['ekf'],
                data['imu']
            )

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

            rospy.loginfo('Tracking boxes...')
            self.tracker.step(sequence)

            # fig, ax = plt.subplots()
            # ax.imshow(data['rgb'])
            # for crop in sequence.scenes[-1].crop_group.crops:
            #     track = crop.crop_box.data
            #     ax.add_patch(
            #         patches.Rectangle(
            #         (track[0], track[1]),
            #         track[2] - track[0],
            #         track[3] - track[1],
            #         linewidth=3,
            #         edgecolor=get_color_from_cluster(int(crop.cluster)),
            #         facecolor='none')
            #     )

            # if len(sequence.scenes) > 1:
            #     for crop in sequence.scenes[-2].crop_group.crops:
            #         track = crop.crop_box.data
            #         offset = crop.estimated_motion_2d
            #         ax.add_patch(
            #             patches.Rectangle(
            #             (track[0] + offset[0], track[1] + offset[1]),
            #             track[2] + offset[0] - track[0],
            #             track[3] + offset[1] - track[1],
            #             linewidth=3,
            #             edgecolor='#FF0000',
            #             facecolor='none')
            #         )
            # detections.mask_group.plot(0.5)
            # plt.show()
            # plt.savefig(
            #     '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/log/'
            #     + str(data['index'])
            #     + '.png'
            # )
            plt.close()

            if data['index'] != 520 and data['index'] % see_sequence == 0:
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

# ----------------------------------------------------------
# v_3d.data.clear()
# scene.plot(
#     data_plot=v_3d.data,
#     line_scalars=np.linspace(-0.5, 0.5, 100),
#     # plane_scalars=(
#     #     np.linspace(-500, 600, 100),
#     #     np.linspace(0, 1300, 100)
#     # ),
#     plot_3d_points_crop=True,
#     plot_emerging_points=True,
#     plot_3d_points_plane=False
# )
# v_3d.show()



# rospy.loginfo('Plotting 2D image with masks...')
# v_2d = Visualizer2D()
# v_2d.plot_mask_group(
#     mask_group,
#     data['rgb'],
#     None,
#     # 'r--',
#     # 'b.',
#     0.5
# )
# v_2d.show()
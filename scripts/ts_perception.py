import signal
import sys

from cv_bridge import CvBridge
import numpy as np
import rospy
import rospkg

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ts_semantic_feature_detector.features_2d.masks import MaskGroup
from ts_semantic_feature_detector.features_3d.camera import StereoCamera
from ts_semantic_feature_detector.features_3d.crop import CornCropGroup
from ts_semantic_feature_detector.features_3d.ground_plane import GroundPlane
from ts_semantic_feature_detector.features_3d.scene import AgriculturalScene
from ts_semantic_feature_detector.features_3d.sequence import AgriculturalSequence
from ts_semantic_feature_detector.input_utils.extrinsics import get_extrinsics
from ts_semantic_feature_detector.input_utils.loaders.sync_loader import SynchronizedLoader
from ts_semantic_feature_detector.segmentation_model.model.mask_rcnn_stem_segmentation import MaskRCNNStemSegmentationModel
from ts_semantic_feature_detector.segmentation_model.ts_dataset.ts_load_dataset import TerraSentiaDataset
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
        self.camera = StereoCamera([fx, fy, cx, cy])

        rospy.loginfo('Getting started...')
        self.main()

    def get_agricultural_scene(
        self,
        img_rgb,
        img_depth,
        camera
    ):
        __, __, masks, scores = self.model.inference(img_rgb)

        mask_group = MaskGroup(masks, scores, binary_threshold=0.5)
        mask_group.metric_filtering('score', score_threshold=0.5)
        mask_group.filter_redundancy(x_coordinate_threshold=20)

        if not mask_group.masks:
            return None

        gp = GroundPlane(
            img_rgb,
            'threshold_gaussian',
            camera,
            img_depth,
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

        crop_group = CornCropGroup(
            mask_group,
            camera,
            img_depth,
            mask_filter_threshold=2,
            ground_plane=gp
        )

        # crop_group.plot_depth_histograms(img_rgb, img_depth)

        return AgriculturalScene(crop_group, gp)
    
    def main(self):
        sequence = AgriculturalSequence()
        v_3d = Visualizer3D()
        
        positions = []
        
        i = 0
        see_sequence = 5

        skip = 0
        for data in self.sync_loader.get_sync_data():
            if skip < 500:
                skip += 1
                continue

            rospy.loginfo('Getting agricultural scene...')
            scene = self.get_agricultural_scene(
                data['rgb'],
                data['depth'],
                self.camera
            )

            if not scene:
                continue

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

            rospy.loginfo('Getting extrinsics...')
            position, orientation = get_extrinsics(
                data['ekf'],
                data['imu']
            )
            positions.append(position)
            scene.add_extrinsics_information(position, orientation)

            sequence.add_scene(scene)
            if i % see_sequence == 0:
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
                    plot_3d_points_plane=False
                )

                n_positions = np.array(positions)
                print(n_positions)
                # v_3d.data.append(go.Scatter3d(
                #         x=[position[0]],
                #         y=[position[1]],
                #         z=[position[2]],
                #         marker = go.scatter3d.Marker(size=5),
                #         opacity=1,
                #         mode='markers'
                # ))
                    
                v_3d.show()
            i += 1

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

    def signal_handler(self, sig, frame):
        print('Exiting...')
        sys.exit(0)

if __name__ == '__main__':
    TerraSentiaPerception()
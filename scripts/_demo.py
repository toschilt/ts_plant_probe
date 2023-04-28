import numpy as np

from ts_semantic_feature_detector.segmentation_model.ts_dataset import ts_load_dataset
from ts_semantic_feature_detector.segmentation_model.model import mask_rcnn_stem_segmentation
from ts_semantic_feature_detector.features_2d.detections import DetectionGroup 
from ts_semantic_feature_detector.features_3d.camera import StereoCamera
from ts_semantic_feature_detector.features_3d.crop import CornCropGroup
from ts_semantic_feature_detector.features_3d.ground_plane import GroundPlane
from ts_semantic_feature_detector.features_3d.scene import AgriculturalScene
from ts_semantic_feature_detector.visualization.visualizer_2d import Visualizer2D
from ts_semantic_feature_detector.visualization.visualizer_3d import Visualizer3D
from ts_semantic_feature_detector.segmentation_model.visualization.log import MetricsVisualizer

import matplotlib.pyplot as plt

dataset = ts_load_dataset.TerraSentiaDataset(
    './ts_semantic_feature_detector/segmentation_model/ts_dataset/data/PNGImages',
    './ts_semantic_feature_detector/segmentation_model/ts_dataset/data/StemPlantMasks',
    mean=[0.3470, 0.4711, 0.3395],
    std_dev=[0.2194, 0.2355, 0.2541],
)

model = mask_rcnn_stem_segmentation.MaskRCNNStemSegmentationModel(
    dataset,
    450,
    800,
    model_path='/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/checkpoints/model_better_mAP_60'
)

# fx = 527.0302734375
# fy = 527.0302734375
# cx = 627.5240478515625
# cy = 341.2162170410156
# camera = StereoCamera([fx, fy, cx, cy])

# depth_img = camera.load_image(
#     '/home/daslab/Documents/dev/catkin_ws/depth.png'
# )

# model.train(0.85, 4, 4, 1000, '../log', '../checkpoints')
rgb_img, ground_data, stem_data = model.inference(
    inference_img_path='/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/data/svo/rgb/1663689704_n0839967000.png'
)

for mask in ground_data['masks']:
    mask = np.moveaxis(mask, 0, 2)
    plt.figure()
    plt.imshow(mask)
    plt.show()

# detections = DetectionGroup(
#     stem_data.boxes,
#     stem_data.masks,
#     stem_data.scores,
#     binary_threshold=0.5
# )
# detections.metric_filtering('score', score_threshold=0.5)
# detections.filter_redundancy(x_coordinate_threshold=20)

# visualizer_2d = Visualizer2D()
# plt.figure()
# visualizer_2d.plot_mask_group(
#     detections.mask_group, 
#     rgb_img,
#     opt_avg_curve='r--',
#     opt_ransac_line='b--',
#     alpha=0.5)
# plt.show()

# visualizer_3d = Visualizer3D()
# gp = GroundPlane(
#     rgb_img,
#     'threshold_gaussian',
#     camera,
#     depth_img)

# gp.plot(
#     visualizer_3d.data,
#     plot_3d_points=False,
#     plot_plan_scalars=(
#         np.linspace(-500, 600, 100),
#         np.linspace(0, 1300, 100)
#     )
# )

# crop_group = CornCropGroup(
#     mask_group,
#     camera,
#     depth_img,
#     mask_filter_threshold=70,
#     ground_plane=gp
# )

# crop_group.plot(
#     visualizer_3d.data,
#     plot_3d_points=True,
#     line_scalars=np.linspace(-200, 200, 100),
#     plot_emerging_point=True,
# )
# visualizer_3d.show()
# visualizer_2d.show(True)

# scene = AgriculturalScene(crop_group, gp)

# scene.add_extrinsics_information(
#     [2.0, 3.0, 1.0],
#     [2.0, 3.0, 1.0]
# )

# m_vis = MetricsVisualizer(
#     '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/src/log/train_log.json',
#     '/home/daslab/Documents/dev/catkin_ws/src/ts_semantic_feature_detector/src/log/validation_log.json'
# )
# m_vis.plot_training_loss_metrics_together()
# m_vis.plot_testing_metrics_comparatively('segm')
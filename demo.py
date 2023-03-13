import numpy as np

from segmentation_model.ts_dataset import ts_load_dataset
from segmentation_model.model import mask_rcnn_stem_segmentation
from features_2d.masks import MaskGroup
from features_3d.camera import StereoCamera
from features_3d.crop import CornCropGroup
from features_3d.ground_plane import GroundPlane
from features_3d.scene import AgriculturalScene
from visualization.visualizer_2d import Visualizer2D
from visualization.visualizer_3d import Visualizer3D

import matplotlib.pyplot as plt

dataset = ts_load_dataset.TerraSentiaDataset(
    './segmentation_model/ts_dataset/data/PNGImages',
    './segmentation_model/ts_dataset/data/StemPlantMasks',
    mean=[0.3618, 0.4979, 0.3245],
    std_dev=[0.1823, 0.1965, 0.2086],
)

model = mask_rcnn_stem_segmentation.MaskRCNNStemSegmentationModel(
    dataset,
    450,
    800,
    model_path='./segmentation_model/model/checkpoints/model_better_mAP_367'
)

fx = 527.0302734375
fy = 527.0302734375
cx = 627.5240478515625
cy = 341.2162170410156
camera = StereoCamera([fx, fy, cx, cy])

depth_img = camera.load_image(
    '/home/daslab/Documents/dev/slam_dataset/utils/extracted/depth000650.png'
)

# model.train(0.85, 4, 4, 50, './log')
rgb_img, __, masks, scores = model.inference(
    '/home/daslab/Documents/dev/slam_dataset/utils/extracted/left000650.png'
)

mask_group = MaskGroup(masks, scores, binary_threshold=0.5)
mask_group.metric_filtering('score', score_threshold=0.5)
mask_group.filter_redundancy(x_coordinate_threshold=20)

# visualizer_2d = Visualizer2D()
# visualizer_2d.plot_mask_group(
#     mask_group, 
#     rgb_img,
#     opt_avg_curve='r--',
#     opt_ransac_line='b--',
#     alpha=0.5)

# visualizer_3d = Visualizer3D()
gp = GroundPlane(
    rgb_img,
    'threshold_gaussian',
    camera,
    depth_img)

# gp.plot(
#     visualizer_3d.data,
#     plot_3d_points=False,
#     plot_plan_scalars=(
#         np.linspace(-500, 600, 100),
#         np.linspace(0, 1300, 100)
#     )
# )

crop_group = CornCropGroup(
    mask_group,
    camera,
    depth_img,
    mask_filter_threshold=70,
    ground_plane=gp
)

# crop_group.plot(
#     visualizer_3d.data,
#     plot_3d_points=True,
#     line_scalars=np.linspace(-200, 200, 100),
#     plot_emerging_point=True,
# )
# visualizer_3d.show()
# visualizer_2d.show(True)

AgriculturalScene(crop_group, gp)
import numpy as np
from PIL import Image

from inference import get_data_from_inference, filter_data_by_score_threshold
# from extract_masks import get_binary_mask, combine_masks
from filter_masks import filter_masks_redundancy
import utils_3d

# import open3d as o3d

def get_3d_points_from_masks(crops, depth_img, K_inv, filter_depth_=True, plot_filter_histogram=False, rgb_img=None):
    crops_3d = []
    for crop in crops:
        #Getting images and preparing data to 3D projection
        binary_mask = crop['binary_mask']
        indices = crop['indices']
        
        crops_3d.append(utils_3d.get_3d_points(indices, binary_mask, depth_img, K_inv, filter_depth_, plot_filter_histogram, rgb_img))
        
    return crops_3d

#Filters the masks for redundancy before getting the 3d points
def get_3d_crops(rgb_img, depth_img, masks, K_inv):
    crop_masks = filter_masks_redundancy(masks, threshold_1d=20)
    return get_3d_points_from_masks(crop_masks, depth_img, K_inv, filter_depth_=True, plot_filter_histogram=False, rgb_img=rgb_img)

#Line in 3D is defined by point + vector*scalars.
#Returns (point, vector)
def get_3d_crops_lines_coefs(crops_3d):
    crops_line_coefs = []
    for crop_3d in crops_3d:
        pc_vector = utils_3d.get_PCA_component(1, crop_3d, debug=False)
        mean_point = np.average(crop_3d, axis=0)

        crops_line_coefs.append([mean_point, pc_vector])

    return crops_line_coefs

def get_3d_crops_lines(crops_line_coefs, scalars):
    scalars = scalars[:, None]
    crops_lines = []
    for coefs in crops_line_coefs:        
        crops_lines.append(coefs[0] + scalars*coefs[1])
    
    return crops_lines

if __name__ == '__main__':
    model_path = "models/model_better_mAP_367"

    base_path = '/home/daslab/Documents/dev/slam_dataset/utils/extracted/'
    depth_base_filename = 'depth00'
    rgb_base_filename = 'left00'
    img_number = '0650'

    depth_path = base_path + depth_base_filename + img_number + '.png'
    rgb_path = base_path + rgb_base_filename + img_number + '.png'

    rgb_img = np.array(Image.open(rgb_path).convert("RGB"))
    depth_img = np.array(Image.open(depth_path))

    boxes, masks, scores = get_data_from_inference(model_path, rgb_path, debug=False)
    best_score_boxes, best_score_masks = filter_data_by_score_threshold(boxes, masks, scores, score_threshold=0.5)
    # plot_img_with_mask(rgb_img, get_binary_mask(combine_masks(best_score_masks), 0.5))

    fx = 527.0302734375
    fy = 527.0302734375
    cx = 627.5240478515625
    cy = 341.2162170410156
    K, K_inv = utils_3d.get_intrinsics_matrices(fx, fy, cx, cy)
    
    crops_3d = get_3d_crops(best_score_masks, depth_img, K_inv)
    lines_coefs = get_3d_crops_lines_coefs(crops_3d)
    lines = get_3d_crops_lines(lines_coefs, np.linspace(-100, 100, 1000))
    utils_3d.plot_3d_objects(crops_3d + lines)

# -----------------------------------------------------------------------
# I tried:
# for crop in crops:
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(crop['points_camera_frame'])
    
    # pcd2 = o3d.geometry.PointCloud()
    # translated_pcd = crop['points_camera_frame']
    # translated_pcd[:, 0] += 500
    # pcd2.points = o3d.utility.Vector3dVector(translated_pcd)
    # pcd2 = pcd2.remove_statistical_outlier(200, 0.5)[0]
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.add_geometry(pcd2)
    # vis.run()
    # vis.destroy_window()

    # filtered_crop = np.asarray(pcd2.points)

    # Tried RANSAC, but does not work very well
    # ransac = linear_model.RANSACRegressor(min_samples=2, residual_threshold=30.0)
    # X_reshaped = filtered_crop[:, -1].reshape(-1, 1) #z values
    # y_reshaped = filtered_crop[:, 0:2].reshape(-1, 2) #x, y values
    # ransac.fit(X_reshaped, y_reshaped)
    # #x_aval = np.linspace(np.min(filtered_crop[:, 0]), np.max(filtered_crop[:, 0]))
    # #y_aval = np.linspace(np.min(filtered_crop[:, 1]), np.max(filtered_crop[:, 1]))
    # z_aval = np.linspace(np.min(filtered_crop[:, 2]), np.max(filtered_crop[:, 2]))

    # y_pred = ransac.predict(np.append(x_aval[:, None], y_aval[:, None], axis=1).reshape(-1, 2))
    # y_pred = ransac.predict(z_aval.reshape(-1, 2))
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from inference import get_data_from_inference, filter_data_by_score_threshold
from extract_3d_crops import get_3d_crops, get_3d_crops_lines_coefs, get_3d_crops_lines
from extract_3d_ground import get_ground_plane
from extract_masks import plot_img_with_mask, combine_masks, get_binary_mask
import utils_3d

if __name__ == '__main__':
    model_path = "models/model_better_mAP_367"

    base_path = '/home/daslab/Documents/dev/slam_dataset/utils/extracted/'
    depth_base_filename = 'depth00'
    rgb_base_filename = 'left00'
    img_number = '0676'

    depth_path = base_path + depth_base_filename + img_number + '.png'
    rgb_path = base_path + rgb_base_filename + img_number + '.png'

    rgb_img = np.array(Image.open(rgb_path).convert("RGB"))
    depth_img = np.array(Image.open(depth_path))
    
    fx = 527.0302734375
    fy = 527.0302734375
    cx = 627.5240478515625
    cy = 341.2162170410156
    _, K_inv = utils_3d.get_intrinsics_matrices(fx, fy, cx, cy)

    #Get ground plane
    threshold_values = {'hLow': 0, 'sLow': 0, 'vLow': 0, 'sHigh': 30,'hHigh': 146, 'vHigh': 199}
    gaussian_filter_size = [12]
    ground_plane = get_ground_plane(rgb_img, depth_img, K_inv, threshold_values, gaussian_filter_size)

    #Get crop lines
    boxes, masks, scores = get_data_from_inference(model_path, rgb_path, debug=False)
    best_score_boxes, best_score_masks = filter_data_by_score_threshold(boxes, masks, scores, score_threshold=0.5)
    crops_3d = get_3d_crops(rgb_img, depth_img, best_score_masks, K_inv)
    crops_lines_coefs = get_3d_crops_lines_coefs(crops_3d)
    crops_lines = get_3d_crops_lines(crops_lines_coefs, np.linspace(-500, 500, 100))

    normal_vector_plane = ground_plane['normal_vector']
    point_plane = ground_plane['mean_point']
    emerging_points = []
    for line_coef in crops_lines_coefs:
        line_point = line_coef[0]
        line_vector = line_coef[1]

        scalar = np.dot((point_plane - line_point), normal_vector_plane)/np.dot(normal_vector_plane, line_vector)
        emerging_points.append(line_point + scalar*line_vector)

    objects_3d = [ground_plane['points_3d']] + crops_3d + crops_lines
    
    data = []
    for object in objects_3d:
        data.append(go.Scatter3d(
        x=object[:, 0], 
        y=object[:, 1], 
        z=object[:, 2], 
        marker=go.scatter3d.Marker(size=3), 
        opacity=0.8, 
        mode='markers'))

    plane_coefs = ground_plane['coefs']
    x = np.linspace(-2000, 2000, 100)
    z = np.linspace(0, 5000, 100)
    x, z = np.meshgrid(x, z)
    y = -(plane_coefs[0]*x + plane_coefs[2]*z + plane_coefs[3])/plane_coefs[1]

    #Plot the calculated ground plane
    data.append(go.Surface(z=z, x=x, y=y))

    for point in emerging_points:
        data.append(go.Scatter3d(
        x=[point[0]], 
        y=[point[1]], 
        z=[point[2]], 
        marker=go.scatter3d.Marker(size=10), 
        opacity=0.8, 
        mode='markers'))
    
    fig = go.Figure(data=data)
    fig.show()

    plot_img_with_mask(rgb_img, get_binary_mask(combine_masks(best_score_masks), 0.5))
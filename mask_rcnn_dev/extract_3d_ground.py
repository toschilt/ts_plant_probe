from functools import partial
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2

import utils_3d

#Used for manually finding values for isolating the ground
def threshold_trackbars(name, threshold_values, val):
    threshold_values[name] = val

#Used for manually finding values for isolating the ground
def gaussian_trackbar(gaussian_filter_size, val):
    gaussian_filter_size[0] = val

#Uses color thresholding and gaussian blur to find the ground mask from a HSV image
def get_ground_binary_mask(hsv_image, threshold_values, gaussian_filter_size):
    gaussian_filter = (2*gaussian_filter_size[0] + 1, 2*gaussian_filter_size[0] + 1)
    gaussian_img = cv2.GaussianBlur(hsv_image, gaussian_filter, 0)

    lower_color_bounds = np.array((threshold_values['hLow'], threshold_values['sLow'], threshold_values['vLow']))
    higher_color_bounds = np.array((threshold_values['hHigh'], threshold_values['sHigh'], threshold_values['vHigh']))
    return cv2.inRange(gaussian_img, lower_color_bounds, higher_color_bounds)

#Used for manually finding values for isolating the ground
#threshold_values is a dictionary with this format: {'hLow': 0, 'sLow': 0, 'vLow': 0, 'sHigh': 30,'hHigh': 146, 'vHigh': 199}
#gaussian_filter_size needs to be a list of a single number (Python reference issues)
def tune_isolating_ground(hsv_image, threshold_values, gaussian_filter_size):
    cv2.namedWindow('control', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    for key in threshold_values.keys():
        cv2.createTrackbar(key, 'control', threshold_values[key], 255, partial(threshold_trackbars, key, threshold_values))
    cv2.createTrackbar('gaussian_filter_size', 'control', gaussian_filter_size[0], 100, partial(gaussian_trackbar, gaussian_filter_size))

    while True:
        binary_mask = get_ground_binary_mask(hsv_image, threshold_values, gaussian_filter_size)

        cv2.imshow('image', binary_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

#Plane equation considered: ax + by + cz + d = 0 - coefficients are [a, b, c, d]
def get_ground_plane_data(binary_mask, depth_img, K_inv):
    indices = np.argwhere(binary_mask != 0)
    points_3d = utils_3d.get_3d_points(indices, binary_mask, depth_img, K_inv, filter_depth_=False, plot_filter_histogram=False)
    
    pca_vectors = utils_3d.get_PCA_components(2, points_3d, debug=False)
    normal_vector = np.cross(pca_vectors[0], pca_vectors[1])
    
    mean_point = np.average(points_3d, axis=0)

    #ax + by + cz + d = 0
    ground_plane_coefs = [normal_vector[0],
                          normal_vector[1],
                          normal_vector[2],
                          -np.sum(normal_vector*mean_point)]
    
    return {'points_3d': points_3d,
            'pca_vectors': pca_vectors,
            'normal_vector': normal_vector,
            'mean_point': mean_point,
            'coefs': ground_plane_coefs}

#ground_plane is the dictionary returned by get_ground_plane_data
#TODO: generalize scalars for lines and plane visualization
def visualize_ground_plane(ground_plane, plot_pc=True, plot_vectors=True):
    scalars = np.linspace(-100, 100, 1000)[:, None]
    pca_lines = [ground_plane['mean_point'] + scalars*pca_vector for pca_vector in ground_plane['pca_vectors']]
    normal_line = [ground_plane['mean_point'] + scalars*ground_plane['normal_vector']]

    plane_coefs = ground_plane['coefs']
    x = np.linspace(-2000, 2000, 100)
    z = np.linspace(0, 5000, 100)
    x, z = np.meshgrid(x, z)
    y = -(plane_coefs[0]*x + plane_coefs[2]*z + plane_coefs[3])/plane_coefs[1]
    plane = np.array([x, y, z]).T

    #TODO: generalize plot_3d_objects from utils_3d
    objects_3d = []
    if plot_pc:
        objects_3d += [ground_plane['points_3d']]
    
    if plot_vectors:
        objects_3d += pca_lines
        objects_3d += normal_line

    data = []
    for object in objects_3d:
        data.append(go.Scatter3d(
        x=object[:, 0], 
        y=object[:, 1], 
        z=object[:, 2], 
        marker=go.scatter3d.Marker(size=3), 
        opacity=0.8, 
        mode='markers'))

    #Plot the calculated ground plane
    data.append(go.Surface(z=z, x=x, y=y))
    
    fig = go.Figure(data=data)
    fig.show()

def get_ground_plane(rgb_img, depth_img, K_inv, threshold_values, gaussian_filter_size):
    hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    binary_mask = get_ground_binary_mask(hsv_image, threshold_values, gaussian_filter_size)
    return get_ground_plane_data(binary_mask, depth_img, K_inv)

if __name__ == '__main__':
    base_path = '/home/daslab/Documents/dev/slam_dataset/utils/extracted/'
    depth_base_filename = 'depth00'
    rgb_base_filename = 'left00'
    img_number = '0652'

    depth_path = base_path + depth_base_filename + img_number + '.png'
    rgb_path = base_path + rgb_base_filename + img_number + '.png'

    rgb_img = np.array(Image.open(rgb_path).convert("RGB"))
    depth_img = np.array(Image.open(depth_path))
    hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

    fx = 527.0302734375
    fy = 527.0302734375
    cx = 627.5240478515625
    cy = 341.2162170410156
    _, K_inv = utils_3d.get_intrinsics_matrices(fx, fy, cx, cy)

    threshold_values = {'hLow': 0, 'sLow': 0, 'vLow': 0, 'sHigh': 30,'hHigh': 146, 'vHigh': 199}
    gaussian_filter_size = [12]
    #Use this function to tune the ground identification parameters
    # tune_isolating_ground(hsv_image, threshold_values, gaussian_filter_size)
    
    binary_mask = get_ground_binary_mask(hsv_image, threshold_values, gaussian_filter_size)
    ground_plane = get_ground_plane_data(binary_mask, depth_img, K_inv)
    visualize_ground_plane(ground_plane)
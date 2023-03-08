import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import plotly.graph_objects as go
from utils import show_maximized

def plot_3d_objects(objects_3d):
    data = []
    for object in objects_3d:
        data.append(go.Scatter3d(
        x=object[:, 0], 
        y=object[:, 1], 
        z=object[:, 2], 
        marker=go.scatter3d.Marker(size=3), 
        opacity=0.8, 
        mode='markers'))

    fig = go.Figure(data=data)
    fig.show()

def get_intrinsics_matrices(fx, fy, cx, cy):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    return K, K_inv

def filter_depth(depth, num_padding_hist=1, num_hist_bins=100, hist_derivative_threshold=70, debug=False):
    #Get the depth histogram
    hist, bins = np.histogram(depth, bins=np.linspace(np.min(depth) - num_padding_hist, np.max(depth) + num_padding_hist, num_hist_bins))

    #Find the distance that has the most occurrences
    most_prob_z_bin_idx = np.argmax(hist)

    #Compute histogram derivative and find the points where it is high
    hist_derivative = np.abs(hist[1:] - hist[:-1])
    filtered_hist_derivative = hist_derivative > hist_derivative_threshold
    
    #Find the extremities of the occurence distribution
    rising_idx = np.where(~filtered_hist_derivative & np.roll(filtered_hist_derivative,-1))[0]
    rising_idx = rising_idx[rising_idx <= most_prob_z_bin_idx]
    falling_idx = np.where(~np.roll(filtered_hist_derivative,-1) & filtered_hist_derivative)[0]
    falling_idx = falling_idx[falling_idx >= most_prob_z_bin_idx]
    
    if rising_idx.any():
        lower_idx_value = rising_idx[-1]
    else:
        lower_idx_value = 0

    if falling_idx.any():
        higher_idx_value = falling_idx[0]
    else:
        higher_idx_value = len(bins) - 3

    #Use the depth at the extremities to clip the depth values
    lower_z = bins[lower_idx_value + 2]
    high_z = bins[higher_idx_value + 2]
    depth = np.clip(depth, lower_z, high_z)

    if debug:
        print(hist)
        print(bins)
        print(hist_derivative)
        print(most_prob_z_bin_idx)
        print(filtered_hist_derivative)
        print(rising_idx)
        print(falling_idx)
        print(lower_idx_value)
        print(higher_idx_value)
        print(lower_z)
        print(high_z)

    return depth, hist, bins

def plot_image_with_mask_and_histogram(rgb_img, binary_mask, hist, bins):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Mask analysed')
    plt.imshow(rgb_img)
    plt.imshow(np.ma.masked_where(binary_mask == 0, binary_mask), alpha=0.7, cmap='plasma')
    plt.subplot(1, 2, 2)
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")
    plt.xlabel('Depth (mm)')
    plt.ylabel('Occurrences')
    show_maximized()

#If plot_filter_histogram is True, rgb_img needs to be informed
def get_3d_points(indices, binary_mask, depth_img, K_inv, filter_depth_=True, plot_filter_histogram=False, rgb_img=None):
    xy = np.array(indices)
    xy[:, [0, 1]] = xy[:, [1, 0]]
    points_2d = np.hstack((xy, np.ones((indices.shape[0], 1))))

    depth_masked_img = np.where(binary_mask != 0, depth_img, 0)
    depth = depth_masked_img[indices[:, 0], indices[:, 1]]
    if filter_depth_:
        depth, hist, bins = filter_depth(depth)

        if plot_filter_histogram:
            plot_image_with_mask_and_histogram(rgb_img, binary_mask, hist, bins)
    
    #point_2d is in homogeneous coordinates, but point_3d is not. The scalar z makes the operation correct.
    points_3d = []
    for z, point_2d in zip(depth, points_2d):
        points_3d.append(z*(K_inv @ point_2d))

    return np.array(points_3d)

def get_PCA_component(n_component, data_3d, debug=False):
    X_reshaped = data_3d.reshape(-1, 3)
    pca = PCA(n_components=n_component)
    pca.fit(X_reshaped)

    if debug:
        print(pca.components_)
        print(pca.singular_values_)

    return pca.components_[n_component-1]

def get_PCA_components(n_components, data_3d, debug=False):
    X_reshaped = data_3d.reshape(-1, 3)
    pca = PCA(n_components=n_components)
    pca.fit(X_reshaped)

    if debug:
        print(pca.components_)
        print(pca.singular_values_)

    return pca.components_
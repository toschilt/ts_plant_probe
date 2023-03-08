import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from extract_masks import get_binary_mask
from inference import get_data_from_inference, filter_data_by_score_threshold
from utils import show_maximized

from sklearn import linear_model

def get_average_X_coordinate_points(indices):
    sorted_indices = np.lexsort(indices[:, [1, 0]].T)
    sorted_xy = indices[sorted_indices, 0:2]
    unique_y, unique_y_idx = np.unique(sorted_xy[:, 0], return_index=True)
    
    x_values = np.split(indices[:, 1], unique_y_idx[1:])
    for i, x in zip(range(0, len(x_values)), x_values):
        x_values[i] = np.average(x)
    average_x = np.array(x_values)

    return average_x, unique_y

#Returns (m, b), m is the angular and b the linear coefficients
def get_line_coefficients_ransac(x_fit, y_fit):
    ransac = linear_model.RANSACRegressor(min_samples=2)
    X_reshaped = x_fit.reshape(-1, 1)
    y_reshaped = y_fit.reshape(-1, 1)
    ransac = ransac.fit(X_reshaped, y_reshaped)

    return (ransac.estimator_.coef_[0][0], ransac.estimator_.intercept_[0])

def get_average_ransac_line_coefficients(indices):
    average_x, unique_y = get_average_X_coordinate_points(indices)
    return get_line_coefficients_ransac(average_x, unique_y)

def evaluate_line_equation_at_y_coord(coefs, y):
    return float(y - coefs[1])/coefs[0]

if __name__ == '__main__':
    model_path = "models/model_better_mAP_367"
    img_path = "data/PNGImages/left006560.png"

    img = Image.open(img_path).convert("RGB")

    #Get data from inference and filter the relevant part of it
    boxes, masks, scores = get_data_from_inference(model_path, img_path, debug=False)
    _, best_score_masks = filter_data_by_score_threshold(boxes, masks, scores, score_threshold=0.5)

    plt.figure()
    plt.imshow(img)
    for mask in masks:
        binary_mask = get_binary_mask(np.moveaxis(mask, 0, 2), pixel_threshold=0.5)
        indices = np.argwhere(binary_mask != 0)

        average_x, unique_y = get_average_X_coordinate_points(indices)
        plt.plot(average_x, unique_y, 'r--')

        coefs = get_line_coefficients_ransac(average_x, unique_y)
        x0 = indices[np.argmin(indices[:, 0]), 1]
        x1 = indices[np.argmax(indices[:, 0]), 1]
        x = np.linspace(x0, x1, 100)
        plt.plot(x, coefs[0]*x + coefs[1], 'y-.')
    show_maximized()
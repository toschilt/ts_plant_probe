import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from inference import get_data_from_inference, filter_data_by_score_threshold
from extract_masks import get_binary_mask
import extract_lines
from utils import show_maximized

#Filter masks checking the bottom_x value. If the different between consecutive bottom_x values is less than the threshold, the masks are merged.
def filter_masks_redundancy(masks, threshold_1d):
    crops = []
    for mask in masks:
        binary_mask = get_binary_mask(np.moveaxis(mask, 0, 2), pixel_threshold=0.5)[:, :, 0] #TODO: generalize pixel threshold
        indices = np.argwhere(binary_mask != 0)

        coefs = extract_lines.get_average_ransac_line_coefficients(indices)
        bottom_x = extract_lines.evaluate_line_equation_at_y_coord(coefs, 720) #TODO: generalize the y-max coord
        crop = {'mask': mask,
                'binary_mask': binary_mask,
                'indices': indices,
                'ransac_coefs': coefs,
                'bottom_x': bottom_x}
        crops.append(crop)

    #Sort the masks by the bottom_x value
    crops.sort(key=lambda crop: crop['bottom_x'])

    #Calculate distance between calculated consecutive bottom points
    bottom_x_values = [crop['bottom_x'] for crop in crops]
    distance_consecutive_bottom_points = np.abs(np.array(bottom_x_values[0:-1]) - np.array(bottom_x_values[1:]))
    distance_idx = (distance_consecutive_bottom_points < threshold_1d).nonzero()[0]
    
    #Iterate over all redudant bottom_x points
    for idx in distance_idx:
        combined_mask = crops[idx]['mask'] + crops[idx + 1]['mask']
        combined_binary_mask = get_binary_mask(np.moveaxis(combined_mask, 0, 2), pixel_threshold=0.5)[:, :, 0] #TODO: generalize pixel threshold
        combined_indices = np.argwhere(combined_binary_mask != 0)
        new_ransac_coefs = extract_lines.get_average_ransac_line_coefficients(combined_indices)
        new_bottom_x = extract_lines.evaluate_line_equation_at_y_coord(new_ransac_coefs, 720) #TODO: generalize the y-max coord
        crops[idx] = {'mask': combined_mask,
                      'binary_mask': combined_binary_mask,
                      'indices': combined_indices,
                      'ransac_coefs': new_ransac_coefs,
                      'bottom_x': new_bottom_x}
        del crops[idx + 1]
        distance_idx -= 1
    
    return crops

def plot_crops(img, crops):
    plt.figure()
    plt.imshow(img)
    plt.xlim(0, 1280) #TODO: generalize the img coords
    plt.ylim(720, 0) #TODO: generalize the img coords
    for crop in crops:
        x0 = crop['bottom_x']
        y1_idx = np.argmin(crop['indices'][:, 0])
        x1 = crop['indices'][y1_idx, 1]

        x_plot = np.linspace(x0, x1, 100)
        y_plot = crop['ransac_coefs'][0]*x_plot + crop['ransac_coefs'][1]

        plt.imshow(np.ma.masked_where(crop['binary_mask'] == 0, crop['binary_mask']), alpha=0.5, cmap='plasma')
        plt.plot(x_plot, y_plot, 'r')

    show_maximized()

if __name__ == '__main__':
    model_path = "models/model_better_mAP_367"
    img_path = "data/PNGImages/left006560.png"

    img = Image.open(img_path).convert("RGB")

    #Get data from inference and filter the relevant part of it
    boxes, masks, scores = get_data_from_inference(model_path, img_path, debug=False)
    _, best_score_masks = filter_data_by_score_threshold(boxes, masks, scores, score_threshold=0.5)

    crops = filter_masks_redundancy(best_score_masks, 20)
    plot_crops(img, crops)
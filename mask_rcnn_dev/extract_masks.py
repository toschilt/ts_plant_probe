import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from inference import get_data_from_inference, filter_data_by_score_threshold, filter_data_by_best_percentage
from utils import show_maximized

def plot_all_masks_separately(num_images_per_side, masks):
    num_masks = masks.shape[0]
    num_rows = int(num_masks/num_images_per_side if num_masks % num_images_per_side == 0 else num_masks/num_images_per_side + 1)

    for i, mask in zip(range(num_masks), masks):
        plt.subplot(num_rows, num_images_per_side, i + 1)
        plt.imshow(np.moveaxis(mask, 0, 2))
    show_maximized()

def combine_masks(masks):
    return np.moveaxis(np.sum(masks, axis=0), 0, 2)

def plot_mask(mask):
    plt.figure()
    plt.imshow(mask)
    plt.show()

def plot_score_and_percentage_masks(score_mask, percentage_mask):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(top=0.96,
                        bottom=0.045,
                        left=0.125,
                        right=0.9,
                        hspace=0.225,
                        wspace=0.2)
    plt.imshow(score_mask)
    plt.title('Score threshold mask')
    plt.subplot(1, 2, 2)
    plt.imshow(percentage_mask)
    plt.title('Percentage threshold mask')
    show_maximized()

    plt.figure()
    plt.imshow(score_mask - percentage_mask)
    plt.title('Difference between score and percentage masks')
    show_maximized()

#Save parameter can be True or False
def plot_image_and_mask_separately(img, mask, save=False):
    plt.figure()
    plt.subplots_adjust(top=0.925,
                        bottom=0.085,
                        left=0.125,
                        right=0.9,
                        hspace=0.35,
                        wspace=0.2)
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.title('Original image')
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.title('Corresponding mask')

    if save:
        plt.savefig('plot_image_and_mask_separately.png', format='png', dpi=600, transparent=True)
    
    plt.show()

def get_binary_mask(mask, pixel_threshold):
    return np.uint8(mask > pixel_threshold)

def plot_img_with_mask(img, mask):
    plt.figure()
    plt.title('Original image and mask combined')
    plt.imshow(img)
    plt.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.7, cmap='plasma')
    show_maximized()

if __name__ == '__main__':
    model_path = "models/model_better_mAP_367"
    img_path = "data/PNGImages/left006530.png"

    img = Image.open(img_path).convert("RGB")
    boxes, masks, scores = get_data_from_inference(model_path, img_path, debug=False)

    score_threshold = 0.5
    best_score_boxes, best_score_masks = filter_data_by_score_threshold(boxes, masks, scores, score_threshold)
    # plot_all_masks_separately(5, best_score_masks)

    # best_data_percentage = 0.9
    # best_percentage_boxes, best_percentage_masks = filter_data_by_best_percentage(boxes, masks, best_data_percentage)
    # plot_all_masks_separately(5, best_percentage_masks)

    score_mask = combine_masks(best_score_masks)
    # plot_mask(score_mask)
    # percentage_mask = combine_masks(best_percentage_masks)
    # plot_score_and_percentage_masks(score_mask, percentage_mask)
    plot_image_and_mask_separately(img, score_mask)
    
    pixel_threshold = 0.5
    binary_score_mask = get_binary_mask(score_mask, pixel_threshold)
    plot_mask(binary_score_mask)

    plot_img_with_mask(img, binary_score_mask)
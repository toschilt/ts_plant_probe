import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from inference import get_data_from_inference, filter_data_by_score_threshold
from extract_masks import combine_masks, get_binary_mask
from utils import show_maximized

from sklearn import linear_model

def average_x_coordinate_mask(indices):
    sortidx = np.lexsort(indices[:, [1, 0]].T)
    sorted_coo = indices[sortidx, 0:2]
    y_values, s = np.unique(sorted_coo[:, 0], return_index=True)
    x_values = np.split(indices[:, 1], s[1:])
    for i, x in zip(range(0, len(x_values)), x_values):
        x_values[i] = np.average(x)
    x_values = np.array(x_values)
    
    return x_values, y_values

#x and y have shape (size, )
def fit_line_ransac(x_fit, y_fit, x_aval):
    ransac = linear_model.RANSACRegressor(min_samples=2)
    X_reshaped = x_fit.reshape(-1, 1)
    y_reshaped = y_fit.reshape(-1, 1)
    ransac.fit(X_reshaped, y_reshaped)
    y_pred = ransac.predict(x_aval.reshape(-1, 1))

    return X_reshaped, y_reshaped, y_pred, ransac

#Type can be: ['polyfit', 'linear_reg', 'ransac', 'average_ransac']
#Plot can be True or False. If True, img is a required argument to plot.
def get_line_fit(masks, pixel_threshold, type, plot=False, img=None):
    if plot:
        plt.figure()
        plt.imshow(img)

    return_items = []
    for mask in masks:
        binary_mask = get_binary_mask(np.moveaxis(mask, 0, 2), pixel_threshold)
        #plt.imshow(np.ma.masked_where(binary_mask == 0, binary_mask), alpha=0.7, cmap='plasma')

        indices = np.argwhere(binary_mask != 0)
        x0 = np.min(indices[:, 1])
        x_max = np.max(indices[:, 1])
        x_aval = np.linspace(x0, x_max)

        if type == 'polyfit':
            # Polynomial Fit
            coef = np.polynomial.polynomial.Polynomial.fit(indices[:, 1], indices[:, 0], 1).convert().coef
            y_pred = coef[0] + x_aval*coef[1]
            if plot:
                plt.title('Polynomial fit using least squares method')
                plt.plot(x_aval[y<720], y_pred[y<720], 'r--')
            
            mask_data = {'mask': mask,
                         'coef': coef,
                         'x_aval': x_aval,
                         'y_pred': y_pred}
            return_items.append(mask_data)      

        if type == 'linear_reg':
            # Sklearn Linear Regression
            X = indices[:, 1].reshape(-1, 1)
            y = indices[:, 0].reshape(-1, 1)

            lr = linear_model.LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(x_aval.reshape(-1, 1))
            if plot:
                plt.plot(x_aval, y_pred, 'm--')
            
            mask_data = {'mask': mask,
                         'lr': lr,
                         'x_aval': x_aval,
                         'y_pred': y_pred}
            return_items.append(mask_data)
        
        if type == 'ransac':
            # Sklearn RANSAC
            X, y, y_pred, ransac = fit_line_ransac(indices[:, 1], indices[:, 0], x_aval)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            if plot:
                plt.plot(x_aval.reshape(-1, 1)[y_pred<720], y_pred[y_pred<720], 'r--')
                plt.scatter(X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".")
                plt.scatter(X[outlier_mask], y[outlier_mask], color="gold", marker=".")

            mask_data = {'mask': mask,
                         'ransac': ransac,
                         'x_aval': x_aval,
                         'y_pred': y_pred}
            return_items.append(mask_data)

        if type == 'average_ransac':
            #Averaging X coordinates for each Y coordinate
            x_values, y_values = average_x_coordinate_mask(indices)
            if plot:
                plt.plot(x_values, y_values, 'r--')

            #Sklearn RANSAC
            X, y, y_pred, ransac = fit_line_ransac(x_values, y_values, x_aval)
            
            if plot:
                plt.plot(x_aval, y_pred, 'm--')

            mask_data = {'mask': mask,
                         'ransac': ransac,
                         'avg_x': x_values,
                         'avg_y': y_values,
                         'x_aval': x_aval,
                         'y_pred': y_pred[:, 0]}
            return_items.append(mask_data)

    if plot:
        plt.savefig('plot_line_fit_' + type + '.png', format='png', dpi=600, transparent=True)
        plt.show()

    return return_items

#TODO: generalize max image y coord
def plot_line_and_tips(x, y, plot=False):
    y_min = np.min(y)
    x_min = x[np.argmin(y)]
    
    y_max = np.max(y)
    x_max = x[np.argmax(y)]
    # if y_max > 719:
    #     y_max = 719
    #     x_max = x[np.abs(y-719).argmin()]
    
    if plot:
        plt.plot(x_min, y_min, 'bx')
        plt.plot(x_max, y_max, 'mx')

        #Line between the max and min points
        plt.plot(x[y < 719], y[y < 719], 'r--')

    return x_min, y_min, x_max, y_max

def get_line_coefficients(x1, y1, x2, y2):
    #y = mx + b
    m = float(y2 - y1)/(x2 - x1)
    b = y1 - m*x1

    return m, b

def get_crop_mask_line(x, y, plot=False):
    x_min, y_min, x_max, y_max = plot_line_and_tips(x, y, plot=plot)
    m, b = get_line_coefficients(x_min, y_min, x_max, y_max)

    return m, b, [x_min, y_min], [x_max, y_max]

#m and b are the line coefficients
def evaluate_line_equation_at_y_coord(m, b, y, plot=False):
    x = float(y - b)/m
    if plot:
        plt.plot(x, y, 'yx')
    return x

#If plot=True, x_min needs to be specified to plot the line
def get_crop_bottom_line(m, b, image_max_y, plot=False, x_min=None):
    bottom_x = evaluate_line_equation_at_y_coord(m, b, image_max_y, plot=plot)
    
    if plot:
        x_line = np.linspace(x_min, bottom_x)
        plt.plot(x_line, x_line*m + b, 'y')

    return bottom_x

def get_all_crops_lines(average_ransac_data, plot_mask_line=False, plot_bottom=False):
    for mask_data in average_ransac_data:
        x = mask_data['x_aval']
        y = mask_data['y_pred']

        m, b, p_min, p_max = get_crop_mask_line(x, y, plot=plot_mask_line)
        bottom_x = get_crop_bottom_line(m, b, 719, plot=plot_bottom, x_min=p_min[0])

        mask_data['bottom_x'] = bottom_x
        mask_data['p_max'] = p_max

    return average_ransac_data

#Only tested for 'average_ransac' return type from get_line_fit function
def filter_crop_line_redundancy(average_ransac_data, pixel_threshold, bottom_x_threshold):
    #Sort the masks by the bottom_x value
    average_ransac_data.sort(key=lambda mask: mask['bottom_x'])
    
    #Calculate distance between calculated consecutive bottom points
    bottom_x_values = [mask['bottom_x'] for mask in average_ransac_data]
    distance_consecutive_bottom_points = np.abs(np.array(bottom_x_values[0:-1]) - np.array(bottom_x_values[1:]))
    distance_idx = (distance_consecutive_bottom_points < bottom_x_threshold).nonzero()[0]
    
    #Iterate over all redudant bottom_x points
    for idx in distance_idx:
        combined_mask = average_ransac_data[idx]['mask'] + average_ransac_data[idx + 1]['mask']
        del average_ransac_data[idx + 1]
        distance_idx -= 1
        average_ransac_data[idx] = get_line_fit([combined_mask], pixel_threshold, 'average_ransac')[0]

    return average_ransac_data

if __name__ == '__main__':
    duplicated_lines = ['left006581.png', 'left006560.png']
    
    # few_points_for_line = ['left006560.png', 'left006581.png', 'left001458.png']
    # points_disposition_problem = ['left006643.png']
    # good_amount_of_points_for_line = ['left001473.png', 'left001431.png']
    
    model_path = "models/model_better_mAP_367"
    img_path = "data/PNGImages/left006560.png"

    img = Image.open(img_path).convert("RGB")

    #Get data from inference and filter the relevant part of it
    boxes, masks, scores = get_data_from_inference(model_path, img_path, debug=False)
    _, best_score_masks = filter_data_by_score_threshold(boxes, masks, scores, score_threshold=0.5)

    #Fitting lines through the masks
    pixel_threshold = 0.5
    average_ransac_data = get_line_fit(best_score_masks, pixel_threshold, 'average_ransac', plot=False, img=img)
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylim(720, 0)
    plt.xlim(0, 1280)
    plt.imshow(img)
    average_ransac_data = get_all_crops_lines(average_ransac_data, plot_mask_line=True, plot_bottom=True)
    #----- To debug lines
    bin_mask = get_binary_mask(combine_masks(best_score_masks), 0.5)
    plt.imshow(np.ma.masked_where(bin_mask == 0, bin_mask), alpha=0.5, cmap='plasma')
    #-----

    plt.subplot(1, 2, 2)
    plt.ylim(720, 0)
    plt.xlim(0, 1280)
    plt.imshow(img)
    bottom_x_threshold = 15
    average_ransac_data = filter_crop_line_redundancy(average_ransac_data, pixel_threshold, bottom_x_threshold)
    average_ransac_data = get_all_crops_lines(average_ransac_data, plot_mask_line=True, plot_bottom=True)
    #----- To debug lines
    merged_masks = np.array([mask_data['mask'] for mask_data in average_ransac_data]) 
    bin_mask = get_binary_mask(combine_masks(merged_masks), 0.5)
    plt.imshow(np.ma.masked_where(bin_mask == 0, bin_mask), alpha=0.5, cmap='plasma')
    #-----
    plt.tight_layout()
    show_maximized()


    # #STILL TESTING -----
    # #Trying an approach to find the emerging points
    # max_mask_points = np.array([mask_data['p_max'] for mask_data in average_ransac_data])
    # center_point = np.where(max_mask_points[:, 0] < 1280/2)[0][-1] 
    # max_mask_points_div = np.split(max_mask_points, [center_point + 1])

    # x_left = np.linspace(0, 1280/2)
    # x_right = np.linspace(1280/2, 1280)
    # # _, _, y_left, _ = fit_line_ransac(max_mask_points_div[0][:, 0], max_mask_points_div[0][:, 1], x_left)
    # # _, _, y_right, _ = fit_line_ransac(max_mask_points_div[1][:, 0], max_mask_points_div[1][:, 1], x_right)
    # # y_left = y_left[:, 0]
    # # y_right = y_right[:, 0]
    # # plt.plot(x_left[y_left < 719], y_left[y_left < 719], 'r--')
    # # plt.plot(x_right[y_right < 719], y_right[y_right < 719], 'b--')

    # coef_left = np.polynomial.polynomial.Polynomial.fit(max_mask_points_div[0][:, 0], max_mask_points_div[0][:, 1], 1).convert().coef
    # coef_right = np.polynomial.polynomial.Polynomial.fit(max_mask_points_div[1][:, 0], max_mask_points_div[1][:, 1], 1).convert().coef

    # plt.plot(x_left, coef_left[0] + x_left*coef_left[1], 'r--')
    # plt.plot(x_right, coef_right[0] + x_right*coef_right[1], 'b--')
    # plt.tight_layout()
    # show_maximized()
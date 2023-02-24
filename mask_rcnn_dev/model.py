from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn_v2

''' 
    Returns the Mask RCNN model with the necessary modifications for fine-tuning

    Args:
        min_size: Resize target for the smaller image side
        max_size: Resize target for the bigger image side
        image_mean: The mean value from all dataset images
        image_std: The standart deviation value from all dataset images
        num_classes: Number of classes desired for the task
'''
def get_model_instance_segmentation(min_size, max_size, image_mean, image_std, num_classes):
    #Load an instance segmentation model pre-trained on COCO
    #model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=min_size, max_size=max_size, image_mean=image_mean, image_std=image_std)

    #Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    #Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
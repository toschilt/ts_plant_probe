import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn

from detection import utils
from detection import transforms as T
from detection.engine import train_one_epoch, evaluate
from data.ts_load_dataset import TerraSentiaFrontalCameraDataset

from matplotlib import pyplot as plt

def get_model_instance_segmentation(num_classes):
    #Load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

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

def get_transform(train_flag):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    
    if train_flag:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ", device)

    num_classes = 2

    dataset = TerraSentiaFrontalCameraDataset("", \
                                              "PNGImages", \
                                              "StemPlantMasks", \
                                              get_transform(train_flag=True))
    
    dataset_test = TerraSentiaFrontalCameraDataset("", \
                                                   "PNGImages", \
                                                   "StemPlantMasks", \
                                                   get_transform(train_flag=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_test = torch.utils.data.Subset(dataset_test, indices)

    data_loader = torch.utils.data.DataLoader(dataset, \
                                              batch_size=5, \
                                              shuffle=True, \
                                              num_workers=4, \
                                              collate_fn=collate_fn)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, \
                                lr=0.005, 
                                momentum=0.9, 
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                   step_size=3, \
                                                   gamma=0.1)

    num_epochs = 100

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader, device=device)
        
        #Save a checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "models/model_" + str(epoch))

    print("Finished training")

if __name__ == "__main__":
    train()
import json
import numpy as np

import torch
from model import get_model_instance_segmentation

from detection.utils import collate_fn
from detection import transforms as T
from detection.engine import train_one_epoch, evaluate

from data.ts_load_dataset import TerraSentiaFrontalCameraDataset

def get_transform(train_flag):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    
    if train_flag:
        transforms.append(T.RandomHorizontalFlip())
        transforms.append(T.RandomPhotometricDistort())

    return T.Compose(transforms)

def train():
    num_classes = 2
    min_size = 450
    max_size = 800

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ", device)

    dataset = TerraSentiaFrontalCameraDataset("", \
                                              "data/PNGImages", \
                                              "data/StemPlantMasks", \
                                              get_transform(train_flag=True))

    mean, std_dev = dataset.get_metrics()
    print('mean =', mean)
    print('std_dev =', std_dev)
    
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    with open("train_dataset.txt", "w") as outfile:
        for idx in train_dataset.indices:
            outfile.write(dataset.get_file_name(idx) + '\n')

    with open("test_dataset.txt", "w") as outfile:
        for idx in test_dataset.indices:
            outfile.write(dataset.get_file_name(idx) + '\n')

    data_loader_train = torch.utils.data.DataLoader(train_dataset, \
                                                    batch_size=4, \
                                                    shuffle=True, \
                                                    num_workers=4, \
                                                    collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(test_dataset, \
                                                   batch_size=4, \
                                                   shuffle=True, \
                                                   num_workers=4, \
                                                   collate_fn=collate_fn)

    model = get_model_instance_segmentation(min_size, max_size, mean, std_dev, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, \
                                lr=0.005, 
                                momentum=0.9, 
                                weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                   step_size=30, \
                                                   gamma=0.1)

    num_epochs = 1000
    checkpoint_epochs = 20
    checkpoint_mAP = 0.01
    last_best_mAP = 0

    train_log = {}
    test_log = {}

    for epoch in range(num_epochs):
        #Training
        train_logger = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()

        #Evaluate
        test_logger = evaluate(model, data_loader_test, device)

        #Getting train loss information
        train_log[epoch] = dict(train_logger.meters)
        for meter in train_log[epoch]:
            train_log[epoch][meter] = train_log[epoch][meter].value
        
        #Getting evaluation metrics
        test_log[epoch] = {}
        test_log[epoch]['bbox'] = list(test_logger.coco_eval['bbox'].stats)
        test_log[epoch]['segm'] = list(test_logger.coco_eval['segm'].stats)
        mAP = np.average(test_logger.coco_eval['segm'].stats[0])
        
        #Save a checkpoint if the model improves by "checkpoint_mAP" or each "checkpoint_epochs" epochs.
        if (mAP - checkpoint_mAP) >= last_best_mAP:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP
            }, "models/model_better_mAP_" + str(epoch))
            last_best_mAP = mAP
        elif epoch % checkpoint_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP
            }, "models/model_safety_checkpoint_" + str(epoch))

        #Write train and test data into a file
        with open("train_log.json", "w") as outfile:
            json.dump(train_log, outfile)

        with open("test_log.json", "w") as outfile:
            json.dump(test_log, outfile)

    print("Finished training")

if __name__ == "__main__":
    train()

import os
import json
from typing import Union, Tuple, List, Dict, Any

import numpy.typing as npt
from PIL import Image
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor, maskrcnn_resnet50_fpn_v2 
from torchvision.transforms import PILToTensor

from ts_semantic_feature_detector.segmentation_model.model.detection import transforms as T
from ts_semantic_feature_detector.segmentation_model.model.detection.engine import train_one_epoch, evaluate
from ts_semantic_feature_detector.segmentation_model.model.detection.utils import collate_fn
from ts_semantic_feature_detector.segmentation_model.ts_dataset import ts_load_dataset

class MaskRCNNStemSegmentationModel:
    """
    Implements the Mask RCNN model to do stem mask segmentation.

    Uses the Improved Mask R-CNN model with a ResNet-50-FPN backbone 
    from the 'Benchmarking Detection Transfer Learning with Vision 
    Transformers' <https://arxiv.org/abs/2111.11429> paper, implemented
    in the PyTorch framework. Versions: torch (1.13.1) and torchvision
    (0.14.1).

    It makes the necessary adaptations to the existing model to change
    the number of classes predicted (in this case, we are interested in
    only one class: the stems). It also implements a training and
    a inference methods to allow easy use.

    #TODO: Remove TerraSentiaDataset object requirement to do inference.
        Create another structure to encapsulate information about the images. 
    #TODO: Add more explanation about train_log and validation_log dictionaries.
    #TODO: Encapsule some of train method routines in private functions.
    #FIXME: This implementation can't load and resume training.
    
    Attributes:
        hyperparams: a dictionary containing the hyperparameters values. More
            description can be found at the constructor method documentation.
        dataset: segmentation_model.ts_dataset.ts_load_dataset.TerraSentiaDataset
            The TerraSentiaDataset object. It contains all images path and
            do all the pre-processing needed. It also contains some information
            about the images, as original size, mean and standard deviation.
        model: torchvision.models.detection.mask_rcnn.MaskRCNN
            The Mask RCNN model customized to the stem segmentation task.
        optimizer: the SGD optimizer for training the model
        lr_scheduler: the LR Scheduler for training the model.
        transforms: a list of PyTorch transforms to be applied to images when loaded.
        train_log: a dictionary containing all the training metrics.
        validation_log: a dictionary containing all the validation metrics.
        last_best_mAP: the last best Average Precision calculated during training.
            It measures how well the model performed at validation set.
        start_epoch: the epoch that the model stopped training.
        train_log: a dictionary containing training metrics data.
        validation_log: a dictionary containing validation metrics data.
        train_dataset_idxs: a list containing the training dataset images.
        validation_dataset_idxs: a list containing the validation dataset images.
    """

    def __init__(
        self,
        ts_dataset: ts_load_dataset.TerraSentiaDataset,
        input_min_size: int,
        input_max_size: int,
        transforms: Any = None,
        model_path: str = None,
        train: bool = False,
        **kwargs: float,
    ):
        """
        Initializes the model with the necessary modifications.

        Args:
            ts_dataset: TerraSentiaDataset object. It is needed for training
                and inference. It encapsules all the data for training and
                has essential information about the images for inference.
            input_min_size: The minimum size to resize image before inserting it
                into the model.
            input_max_size: The maximum size to resize image before inserting it
                into the model.
            transforms: a list of PyTorch transforms to be applied to images when loaded.
                If transforms is not None, it will be set for training or inference
                depending on model_path value.
            model_path: a string containing the path to the trained model. If it is None,
                transforms attribute is evaluated for training. If it is not None, the
                transforms attribute is evaluated for inference.
            train: a boolean value indicating if the user wants to train the network.

        **kwargs:
            lr: the SGD optimizer learning rate. Default = 0.005
            momentum: the SGD optimizer momentum. Default = 0.9 
            weight_decay: the SGD optimizer weight decay. Default = 0.0005
            step_size: the amount of epochs to apply learning rate decay. Default = 30
            gamma: the learning decay rate. Default = 0.1
            checkpoint_epochs: the number of epochs between safety model saves. Default = 20
            checkpoint_mAP_threshold: the minimum mAP improvement to save a new model.
                Default = 0.01
        """
        self.hyperparams = {
            'lr': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'step_size': 100,
            'gamma': 0.1,
            'checkpoint_epochs': 20,
            'checkpoint_mAP_threshold': 0.01,
        }
        for key in kwargs:
            self.hyperparams[key] = kwargs[key]
        
        self.dataset = ts_dataset
        self.model, self.optimizer, self.lr_scheduler = self._get_model(
            input_min_size,
            input_max_size, 
            model_path)

        if transforms is not None:
            self.dataset.transforms = transforms
        elif train:
            self.dataset.transforms = self._get_transforms(training=True)
        else:
            self.dataset.transforms = self._get_transforms(training=False)

    def _get_model(
        self,
        input_min_size: int,
        input_max_size: int,
        model_path: str = None,
        num_classes: int = 2,
    ) -> Tuple[MaskRCNN, torch.optim.SGD, torch.optim.lr_scheduler.StepLR]:
        """
        Constructs the model from PyTorch and adapt to stem segmentation task.

        It also constructs a SGD optimizer and a learning rate scheduler.
        
        Args:
            input_min_size: The minimum size to resize image before inserting it
                into the model.
            input_max_size: The maximum size to resize image before inserting it
                into the model.
            model_path: a string containing the path to the trained model. If
                specified, model weights are loaded.
            num_classes: The number of classes that the model will look for.
                For this task, num_classes should be 2 (stem and background).
                This value is used as default.

        Returns:
            the MaskRCNN model, the SGD optimizer and the LR Scheduler, respectively.
        """
        model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT", 
            min_size=input_min_size, 
            max_size=input_max_size, 
            image_mean=self.dataset.mean, 
            image_std=self.dataset.std_dev)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            num_classes)

        # Get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Using device: ", self.device)
        model.to(self.device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=self.hyperparams['lr'], 
            momentum=self.hyperparams['momentum'], 
            weight_decay=self.hyperparams['weight_decay'])
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hyperparams['step_size'],
            gamma=self.hyperparams['gamma'])
        
        self.last_best_mAP = 0
        self.start_epoch = 0
        self.train_log = {}
        self.validation_log = {}
        self.train_dataset_idxs = None
        self.validation_dataset_idxs = None

        if model_path is not None:
            checkpoint = torch.load(model_path)
            # To continue training, we need to start from the next epoch (+1)
            self.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_best_mAP = checkpoint['mAP']
            self.train_log = checkpoint['train_log']
            self.validation_log = checkpoint['validation_log']
            self.train_dataset_idxs = checkpoint['train_dataset_idxs']
            self.validation_dataset_idxs = checkpoint['validation_dataset_idxs']
        
        return model, optimizer, lr_scheduler
    
    def _get_transforms(
            self,
            training: bool
    ) -> T.Compose:
        """
        Returns standard PyTorch transforms composed for the mask segmentation task.

        Args:
            training: boolean value that indicates if the model will be used
                for training.
        """
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        
        if training:
            transforms.append(T.RandomHorizontalFlip())
            transforms.append(T.RandomPhotometricDistort())

        return T.Compose(transforms)
    
    def _log_to_file(
            self,
            path: str, 
            data: Union[List[Any], Dict[Any, Any]],
        ) -> None:
        """
        Write a file with desired data.

        #TODO: create the folder if it not exist

        Args:
            path: The path where it is desired to write the log.
            data: a list or a dictionary with the data to be logged.
                If a dictionary is passed, it dumps into the file 
                in a json format. If a list is passed, it writes
                every position in a new line.
        """
        with open(path, "w") as fp:
            if isinstance(data, dict):
                json.dump(data, fp)
            else:
                for d in data:
                    fp.write(d)
                    fp.write('\n')

    def _divide_dataset(
        self, 
        train_percentage: float
    ):
        """
        Divide the dataset into training and validation subsets.

        Args:
            train_percentage: the percentage of all the dataset that will be used
                as training images. The remaining portion will be used to evaluate
                the model during training.

        Returns:
            The training and validation subsets, respectivaly. Both are lists with
                the indices that represent their position at the
                ts_load_dataset.TerraSentiaDataset list of images.
        """
        train_size = int(train_percentage * self.dataset.num_imgs)
        test_size = self.dataset.num_imgs - train_size
        training_dataset, validation_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_size, test_size])
        
        return training_dataset, validation_dataset

    def train(
        self,
        train_percentage: float,
        training_batch_size: int,
        validation_batch_size: int,
        num_epochs: int,
        log_path: str = None,
        num_workers: int = 4,
    ) -> None:
        """
        Train the model and log data when needed.

        Args:
            train_percentage: the percentage of all the dataset that will be used
                as training images. The remaining portion will be used to evaluate
                the model during training.
            training_batch_size: the size of the batch size during training.
            validation_batch_size: the size of the batch size during validation.
            num_epochs: number of epochs for the training.
            log_path: the path to the folder where the logs will be written. If None,
                logging is deactivated.
            num_workers: the number of sub-processes to use for data loading.
        """
        if self.train_dataset_idxs is None and self.validation_dataset_idxs is None:
            self.train_dataset_idxs, self.validation_dataset_idxs = self._divide_dataset(train_percentage)
        
        if log_path is not None:
            self._log_to_file(
                os.path.join(log_path, 'train_imgs.log'),
                [self.dataset.png_imgs[idx] for idx in self.train_dataset_idxs.indices],
            )
            self._log_to_file(
                os.path.join(log_path, 'validation_imgs.log'),
                [self.dataset.png_imgs[idx] for idx in self.validation_dataset_idxs.indices],
            )

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset_idxs,
            batch_size=training_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
        
        validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset_idxs,
            batch_size=validation_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
        
        for epoch in range(self.start_epoch, num_epochs):
            train_logger = train_one_epoch(
                self.model,
                self.optimizer,
                train_loader, 
                self.device,
                epoch,
                print_freq=10)
            self.lr_scheduler.step()

            test_logger = evaluate(
                self.model,
                validation_loader,
                self.device)

            # Getting training metrics
            self.train_log[epoch] = dict(train_logger.meters)
            for meter in self.train_log[epoch]:
                self.train_log[epoch][meter] = self.train_log[epoch][meter].value
            
            # Getting evaluation metrics
            self.validation_log[epoch] = {}
            self.validation_log[epoch]['bbox'] = list(test_logger.coco_eval['bbox'].stats)
            self.validation_log[epoch]['segm'] = list(test_logger.coco_eval['segm'].stats)
            mAP = test_logger.coco_eval['segm'].stats[0]
            
            save_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'mAP': mAP,
                'train_log': self.train_log,
                'validation_log': self.validation_log,
                'train_dataset_idxs': self.train_dataset_idxs,
                'validation_dataset_idxs': self.validation_dataset_idxs
            }

            # Save a checkpoint if the model improves by "checkpoint_mAP" or each "checkpoint_epochs" epochs.
            if (mAP - self.hyperparams['checkpoint_mAP_threshold']) >= self.last_best_mAP:
                torch.save(save_data, "models/model_better_mAP_" + str(epoch))
                self.last_best_mAP = mAP
            elif (epoch % self.hyperparams['checkpoint_epochs']) == 0:
                torch.save(save_data, "models/model_safety_checkpoint_" + str(epoch))

            if log_path is not None:
                self._log_to_file(
                    os.path.join(log_path, 'train_log.json'),
                    self.train_log,
                )
                self._log_to_file(
                    os.path.join(log_path, 'validation_log.json'),
                    self.validation_log,
                )

        print("Finished training!")

    def inference(
        self,
        inference_img: npt.ArrayLike = None,
        inference_img_path: str = None
    ) -> Tuple[Image.Image, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Passes a image through the network and outputs the result.

        The result is filtered by project convenience, therefore some
        network outputs are inaccessible using this method.

        #TODO: incorporate the same transformation object from
            the dataset.

        Args:
            inference_img: a Numpy array containing the image. If it is not
                provided, the path to the image must be informed.
            inference_img_path: the path to the image. If it is not provided,
                the path to the image must be informed.

        Returns:
            the RGB image (PIL image class) and
            a tuple of three Numpy arrays: the bounding boxes, 
            the mask images and the scores for each one of 
            the detected instances, respectively.
        """ 
        img = None   
        if inference_img is not None:
            img = Image.fromarray(inference_img).convert("RGB")
        elif inference_img_path is not None:
            img = Image.open(inference_img_path).convert("RGB")

        img_tensor = None
        if torch.cuda.is_available():
            img_tensor = PILToTensor()(img).unsqueeze_(0).cuda()/255
        else:
            img_tensor = PILToTensor()(img).unsqueeze_(0)/255

        self.model.eval()
        img_tensor.to(self.device)
        
        predictions = self.model(img_tensor)
        boxes = predictions[0]['boxes'].detach().cpu().numpy()
        masks = predictions[0]['masks'].detach().cpu().numpy()
        scores = predictions[0]['scores'].detach().cpu().numpy()

        return img, boxes, masks, scores
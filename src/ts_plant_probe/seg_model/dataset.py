"""
TerraSentia custom dataset module.

Implements the TerraSentiaDataset class.

#TODO: fix hint typing of PyTorch transforms
"""
import json
import logging
import os

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import PILToTensor
from typing import Any

class TerraSentiaDataset(torch.utils.data.Dataset):
    """
    TerraSentia dataset custom loader.

    Attributes:
        png_path : str
            a string containing the PNG images folder path.
        mask_path : str
            a string containing the labeled mask images folder path.
        transforms : Any
            a list of PyTorch transforms to be applied to images when loaded.
        png_imgs : list[str]
            a list of all PNG images file paths lexicographically sorted.
        mask_imgs: list[str]
            a list of all labeled mask images file paths lexicographically sorted.
        num_imbs: int
            a integer number representing how many images there are in the dataset.
        image_size: tuple(int, int)
            a tuple containing the width and height of the image, respectively.
        mean: torch.FloatTensor
            a PyTorch tensor containing the RGB mean of all dataset images.
        std_dev: torch.FloatTensor
            a PyTorch tensor containing the RGB standard deviation of all dataset images.
    """

    def __init__(
        self,
        png_path: str,
        mask_path: str,
        transforms: Any = None,
        mean: torch.FloatTensor = None,
        std_dev: torch.FloatTensor = None,
        metrics_path: str = None
    ):
        """
        Initializes the dataset with custom data path and PyTorch transforms.

        Sorts the PNG and masks images according to file names. Calculates the
        mean and standard deviation of all dataset images if mean or std_dev
        are not provided. If only one is provided, the argument is ignored.

        Args:
            png_path: A string containing the PNG images folder path.
            mask_path: A string containing the labeled masks images folder path.
            transforms: A list containing the PyTorch transforms to be applied
                to images when loaded.
            mean: A PyTorch tensor containing the RGB channels mean of all dataset images.
            std_dev: A PyTorch tensor containing the RGB standard deviation of all dataset images.
        """
        self.logger = logging.getLogger(__name__)

        self.png_path = png_path
        self.mask_path = mask_path
        self.mask_class_path = mask_path + '/SegmentationClass/'
        self.mask_obj_path = mask_path + '/SegmentationObject/'
        self.transforms = transforms

        self.logger.info('Loading PNG images from %s', self.png_path)
        self.png_imgs = sorted(os.listdir(self.png_path))
        self.logger.info('Loading mask class images from %s', self.mask_path)
        self.mask_class_imgs = sorted(os.listdir(self.mask_class_path))
        self.logger.info('Loading mask object images from %s', self.mask_path)
        self.mask_obj_imgs = sorted(os.listdir(self.mask_obj_path))

        self.num_imgs = len(self.png_imgs)
        self.logger.info('Found %d images', self.num_imgs)
        self.img_size = Image.open(os.path.join(self.png_path, self.png_imgs[0])).size
        self.logger.info('Image size: %d x %d', self.img_size[0], self.img_size[1])
        
        labelmap_path = os.path.join(self.mask_path, 'labelmap.txt')
        self.logger.info('Loading labelmap from %s', labelmap_path)
        self.labelmap = np.loadtxt(
            labelmap_path,
            dtype=str,
            delimiter=':',
            skiprows=1
        )

        labelmap = []
        for label in self.labelmap[:-1]:
            rgb = label[1].split(',')
            grayscale = 299/1000 * int(rgb[0]) + 587/1000 * int(rgb[1]) + 114/1000 * int(rgb[2])
            labelmap.append([label[0], int(grayscale)])
        self.labelmap = np.array(labelmap)
        
        if mean is None or std_dev is None:
            if metrics_path is not None:
                self.logger.info('Loading RGB mean and standard deviation from %s', metrics_path)
                if os.path.isfile(metrics_path):
                    metrics = json.load(open(metrics_path))
                    self.mean = torch.tensor(metrics['mean'])
                    self.std_dev = torch.tensor(metrics['std_dev'])
                else:
                    self.logger.warning('Metrics file not found. Calculating RGB mean and standard deviation')
                    self.mean, self.std_dev = self._get_metrics()
            else:
                self.logger.info('Calculating RGB mean and standard deviation')
                self.mean, self.std_dev = self._get_metrics()
        else:
            self.logger.info('Using provided RGB mean and standard deviation')
            self.mean = mean
            self.std_dev = std_dev

        if metrics_path is not None:
            self.logger.info('Saving RGB mean and standard deviation to %s', metrics_path)
            metrics = {'mean': self.mean.tolist(), 'std_dev': self.std_dev.tolist()}
            json.dump(metrics, open(metrics_path, 'w+'))

        self.logger.info('RGB mean: %s', self.mean)
        self.logger.info('RGB standard deviation: %s', self.std_dev)

    def __getitem__(self, idx: int):
        """
        Get data about a single dataset frame by index.

        Loads the PNG and labeled images. Also extracts single object instance masks,
        their corresponding bounding boxes and areas. Apply the specified PyTorch
        transformations to the data.

        Args:
            idx: The requested dataset frame index.

        Returns:
            The transformed RGB image and a dictionary (target) containing the following keys:
                boxes: a tensor containing tuples with the corresponding mask box coordinates 
                    in the format (xmin, ymin, xmax, ymax)
                labels: a tensor containing an array with labels for the dataset masks. 
                    In this case, we are only interested to find crop stems, thus this 
                    array is filled with ones.
                masks: a tensor containing a list with a single object instance masks.
                image_id: a tensor containing the provided image index.
                area: a tensor containing an array representing the area of all bounding boxes.
                iscrowd: a tensor containing an array with a flag that indicates if the
                    object is occluded. For this implementation, this array is filled
                    with zeros.
        """
        img_path = os.path.join(self.png_path, self.png_imgs[idx])
        class_mask_path = os.path.join(self.mask_class_path, self.mask_class_imgs[idx])
        obj_mask_path = os.path.join(self.mask_obj_path, self.mask_obj_imgs[idx])

        rgb_img = Image.open(img_path).convert("RGB")
        class_mask = np.asarray(Image.open(class_mask_path).convert('L'))
        obj_mask = np.asarray(Image.open(obj_mask_path).convert('L'))

        masks = np.zeros((1, rgb_img.size[1], rgb_img.size[0]), dtype=np.uint8)
        boxes = []
        labels = []

        class_ids = np.unique(class_mask)
        for c_id in class_ids[1:]:
            # Get the class ID
            id = np.where(self.labelmap[:, 1].astype(np.uint8) == c_id)[0][0]
            # Reserve the first class ID for background
            id += 1

            # Get the class mask
            c_mask = class_mask == c_id

            # Mask the object mask with the class mask
            objs = np.ma.masked_array(obj_mask, ~c_mask)

            # Different instances are encoded as different colors in object mask image
            obj_ids = np.unique(objs)
            # Removes the empty ID
            obj_ids = obj_ids[:-1]
            # Get the binary mask for each object instance
            c_mask = objs == obj_ids[:, None, None]
            masks = np.concatenate((masks, c_mask), axis=0)

            # Get the bounding box corresponding to each object instance mask
            # Append the id for each object instance
            num_objs = len(obj_ids)
            for i in range(num_objs):
                pos = np.where(c_mask[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(id)

        #Drop the first mask, which is empty
        masks = masks[1:, :, :]

        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:,2] - boxes[:, 0])
        total_objs = len(labels)
        iscrowd = torch.zeros((total_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Apply all the requested transformation to the image and its data
        if self.transforms is not None:
            rgb_img, target = self.transforms(rgb_img, target)

        return rgb_img, target

    def __len__(self):
        """Returns the dataset length."""
        return self.num_imgs
    
    def _get_metrics(self):
        """
        Calculate the RGB mean and standard deviation from all images in the dataset.

        This method is called when a TerraSentiaDataset object is created,
        if the mean and the standard deviation are not provided.

        Returns:
            The mean and standard deviation as PyTorch tensors, respectively.
        """
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])

        for img in self.png_imgs:
            img = PILToTensor()(Image.open(os.path.join(self.png_path, img)).convert("RGB"))/255
            psum += img.sum(axis=[1, 2])
            psum_sq += (img ** 2).sum(axis=[1, 2])
        
        count = float(len(self.png_imgs) * self.img_size[0] * self.img_size[1])

        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)

        return total_mean, total_std
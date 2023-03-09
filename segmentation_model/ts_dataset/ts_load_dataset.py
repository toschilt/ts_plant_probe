"""
TerraSentia custom dataset module.

Implements the TerraSentiaDataset class.

#TODO: fix hint typing of PyTorch transforms
"""

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
        std_dev: torch.FloatTensor = None
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
        self.png_path = png_path
        self.mask_path = mask_path
        self.transforms = transforms
        self.png_imgs = sorted(os.listdir(self.png_path))
        self.mask_imgs = sorted(os.listdir(self.mask_path))
        self.num_imgs = len(self.png_imgs)
        self.img_size = Image.open(os.path.join(self.png_path, self.png_imgs[0])).size

        if mean is None or std_dev is None:
            self.mean, self.std_dev = self._get_metrics()
        else:
            self.mean = mean
            self.std_dev = std_dev

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
        mask_path = os.path.join(self.mask_path, self.mask_imgs[idx])

        rgb_img = Image.open(img_path).convert("RGB")
        gray_mask = np.asarray(Image.open(mask_path).convert('L'))

        # Different instances are encoded as different colors in mask image
        obj_ids = np.unique(gray_mask)
        # Removes the background ID
        obj_ids = obj_ids[1:]

        # Get the binary mask for each object instance
        masks = gray_mask == obj_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Get the bounding box corresponding to each object instance mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # There is only one class in this task. Fills the array with ones.
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:,2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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
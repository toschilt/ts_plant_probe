import os
import sys
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from torchvision.transforms import PILToTensor

class TerraSentiaFrontalCameraDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, images_folder, mask_folder, transforms):
        #Set the needed paths
        self.root_folder = root_folder
        self.images_folder = os.path.join(root_folder, images_folder)
        self.mask_folder = os.path.join(root_folder, mask_folder)

        self.transforms = transforms

        #Load all image file names, sorting them to ensure alignment
        self.imgs = sorted(os.listdir(self.images_folder))
        self.masks = sorted(os.listdir(self.mask_folder))

    def __getitem__(self, idx):
        #Get image and its mask path
        img_path = os.path.join(self.images_folder, self.imgs[idx])
        mask_path = os.path.join(self.mask_folder, self.masks[idx])

        #print("img_path: ", img_path)
        #print("mask_path: ", mask_path)

        #Load the image and the mask. The mask is not converted to RGB.
        img = Image.open(img_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path).convert('L'))

        #Instances are encoded as different colors
        obj_ids = np.unique(mask)
        #print("obj_ids: ", obj_ids)
        
        #First ID is the background, removing it
        obj_ids = obj_ids[1:]
        #print("obj_ids_without_bg: ", obj_ids)

        #MASKS DEFINITION
        #Get the binary mask from the colors
        masks = mask == obj_ids[:, None, None]
        #print("masks.shape: ", masks.shape)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        #BOUNDING BOXES DEFINITION
        #Get the bounding boxes coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        #Convert to a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #CLASSES DEFINITION
        #There is only one class
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_metrics(self):
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])

        for img in self.imgs:
            img = PILToTensor()(Image.open(os.path.join(self.images_folder, img)).convert("RGB"))/255
            psum += img.sum(axis=[1, 2])
            psum_sq += (img ** 2).sum(axis=[1, 2])
        
        count = float(len(self.imgs) * 1280 * 720)

        # mean and std
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)

        return total_mean, total_std

    def get_file_name(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return len(self.imgs)
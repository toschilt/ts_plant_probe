"""
Implements Mask-RCNN architecture utilities using the PyTorch framework.

Loads a custom PyTorch dataset with TerraSentia images. Loads the Mask-RCNN 
architecture model with pre-trained COCO weights ready to be trained or used for
inference. Implements training with logging capabilities and inference for other
modules.

Modules:
    ts_dataset: Custom dataset to load TerraSentia images
    mask_rcnn_model: Mask-RCNN model implementation
"""
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torchvision.transforms.functional as F
from torchvision.transforms import PILToTensor
from torchvision.utils import draw_segmentation_masks

from inference import inference

plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

model_path = "models/model_better_mAP_367"
img_path = "data/PNGImages/left006602.png"

predictions = inference(model_path, img_path)
img = Image.open(img_path).convert("RGB")

masks = predictions[0]['masks'].detach().cpu().numpy()
scores = predictions[0]['scores'].detach().cpu().numpy()

selected_masks = np.zeros_like(masks[0])
for i in range(len(masks[:20])):
    selected_masks += masks[i]

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img)
axarr[1].imshow(selected_masks[0])
plt.show()

bool_selected_masks = selected_masks > 0.75
plt.imshow(np.uint8(bool_selected_masks[0]))
plt.show()

show(draw_segmentation_masks(PILToTensor()(img), masks=torch.tensor(bool_selected_masks), alpha=0.7))
plt.show()
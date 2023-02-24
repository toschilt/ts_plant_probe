import torch
from PIL import Image

from torchvision.transforms import PILToTensor
from model import get_model_instance_segmentation

def inference(model_path, img_path):
    num_classes = 2
    min_size = 450
    max_size = 800
    mean = torch.tensor([0.3618, 0.4979, 0.3245])
    std_dev = torch.tensor([0.1823, 0.1965, 0.2086])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ", device)

    model = get_model_instance_segmentation(min_size, max_size, mean, std_dev, num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, \
                                lr=0.005, 
                                momentum=0.9, 
                                weight_decay=0.0005)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']

    model.eval()
    img = Image.open(img_path).convert("RGB")
    img_tensor = PILToTensor()(img).unsqueeze_(0)/255
    return model(img_tensor)

if __name__ == "__main__":
    model_path = "models/model_better_mAP_367"
    img_path = "data/PNGImages/left006602.png"
    inference(model_path, img_path)
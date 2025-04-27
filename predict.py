
import torch
from torchvision import transforms
from PIL import Image
from model import DeepResNet
from config import resize_x, resize_y, input_channels, num_classes

def cryptic_inf_f(model, list_of_image_paths, device):
    # Define the transform (same as test_transform)
    inference_transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model.eval()
    model = model.to(device)

    batch_images = []

    for img_path in list_of_image_paths:
        image = Image.open(img_path).convert("RGB")
        image = inference_transform(image)
        batch_images.append(image)

    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().tolist()

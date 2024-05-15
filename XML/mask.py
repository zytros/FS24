from transformers import ViTForImageClassification
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

def load_transform_image(image_pth, size=224):
    image = Image.open(image_pth).convert("RGB")
    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor

def get_masked_prediction(image_fn, mask):
    image = load_transform_image(image_fn)
    if image.shape != mask.shape:
        return 'Image and mask shape mismatch'
    masked_image = image * mask
    output = model(masked_image).logits.softmax(dim=1)
    top_values, top_indices = output.topk(5)
    return {
        "percentages": top_values.detach().numpy().tolist()[0],
        "classes": top_indices.detach().numpy().tolist()[0]
    }
    
masked_preds = get_masked_prediction('data/cougar_gt.png', torch.rand(1, 3, 224, 224))
print(masked_preds)

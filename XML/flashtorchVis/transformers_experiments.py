from transformers import SwinForImageClassification
import torch
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
import numpy as np
swin = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
backprop = Backprop(swin)

image = load_image('SWIN/data/cat.jpeg')
input_image = apply_transforms(image)
swin.eval()
'''output = swin(input_image).logits
output = torch.nn.functional.softmax(output, dim=1)
output = output.detach().numpy().flatten()
print(np.argmax(output))'''
target_class = 285
backprop.visualize(input_image, target_class, guided=True)
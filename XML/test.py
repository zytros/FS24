import torch
#from pytorch_pretrained_gans import make_gan
from pytorch_pretrained_gans_ import make_gan
import matplotlib.pyplot as plt
import heapq
import numpy as np

G = make_gan(gan_type='biggan')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])
x_img = x[0].detach().cpu().numpy().transpose(1, 2, 0)
plt.imshow(x_img)
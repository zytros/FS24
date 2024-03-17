import torch
from pytorch_pretrained_gans import make_gan

G = make_gan(gan_type='biggan')
y = G.sample_class(batch_size=1)
z = G.sample_latent(batch_size=1)
x = G(z=z, y=y)

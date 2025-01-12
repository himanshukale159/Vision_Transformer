import torch
import torch as nn

from vit import ViT
from utils import get_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cfg = {
    'patch_size' : 16,
    'latent_size' : 768,
    'n_channels' : 3,
    'num_heads' : 12,
    'num_encoders':12,
    'dropout' : 0.1,
    'num_classes' : 10,
    'size' : 224,
    'batch_size' : 4
}

patch_size,latent_size,n_channels,num_heads,num_encoders,num_classes,dropout,size,batch_size = get_config(cfg)

model = ViT()

#testing model
input = torch.randn(4,3,224,224)
output = ViT(input)

print(f"Input size : {input.shape}")
print(f"Output size: {output.shape}")
print(f"Model Checked Successfully")
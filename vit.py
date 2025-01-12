import einops
import torch
from torch import nn

from block import InputEmbedding,EncoderBlock
from utils import get_config

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
device = 'cpu'
patch_size,latent_size,n_channels,num_heads,num_encoders,num_classes,dropout,size,batch_size = get_config(cfg)
print(type(num_encoders))
class ViT(nn.Module):

    def __init__(self,
                 num_encoders = num_encoders,
                 latent_size = latent_size,
                 num_classes = num_classes,
                 dropout = dropout,
                 device = device,
                 ):
        super(ViT,self).__init__()

        self.num_encoders = num_encoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.latent_size = latent_size

        # Embedding Layer
        self.embedding = InputEmbedding()

        # Encoder Stack
        self.enc_stack = nn.ModuleList([EncoderBlock() for i in range(self.num_encoders)])

        # VIT-MLP
        self.MLP = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size,self.latent_size),
            nn.Linear(self.latent_size,self.num_classes)
        )

    def forward(self, test_input):

        # Finding Patch Embeddings
        encoder_output = self.embedding(test_input)

        # Looping through the encoder stack
        for encoder_layer in self.enc_stack:
            encoder_output = encoder_layer(encoder_output)

        # Extracting the class token embedding
        class_token_embedding = encoder_output[:,0]

        # Sending it through the MLP
        output = self.MLP(class_token_embedding)

        return output
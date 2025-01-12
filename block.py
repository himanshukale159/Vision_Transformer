import einops
import torch
from torch import nn
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

class InputEmbedding(nn.Module):

    def __init__(self,
                 patch_size = patch_size,
                 latent_size = latent_size,
                 n_channels = n_channels,
                 batch_size = batch_size,
                 device = device):
        super(InputEmbedding,self).__init__()

        self.patch_size = patch_size
        self.latent_size = latent_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.device = device

        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear Projection Layer
        self.LinearProjection = nn.Linear(self.input_size,
                                       self.latent_size)
        
        # Class Token
        self.ClassToken = nn.Parameter(torch.randn(self.batch_size,
                                                   1,
                                                   self.latent_size))  # Creates a class token vector of size (4,1,768)
        
        # Positional Embedding
        self.PositionalEmbedding = nn.Parameter(torch.randn(self.batch_size,
                                                            1,
                                                            self.latent_size)) # Creates a positional embedding of size (4,1,768)
        
    def forward(self,
                input_data):
        
        # Patchify the input data --> Convert the image of size (224,224,3) to (196,768) [ where 196 = (224*224)/(16*16) Num_patches, 768 = (16*16*3) Flattened vector of every patch 
        patches = einops.rearrange(
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size
            ) 
        
        linear_projection = self.LinearProjection(patches)  # Convert the (4,196,768) to (4,196,768) :: the first 768 is just by chance and depend on patch size later one is latent_dim
        b,n,p = linear_projection.shape

        linear_projection = torch.cat((self.ClassToken,linear_projection),dim=1) # Concatentate classtoken to linear project (4,197,768)
        
        positional_embedding = einops.repeat(self.PositionalEmbedding, 'b 1 d -> b m d', m=n+1) # Converted pos_embed (4,1,768) --> (4,197,768) 

        linear_projection += positional_embedding # Added pos_embed to linear_proj (4,196,768)

        return linear_projection
    

class EncoderBlock(nn.Module):

    def __init__(self,
                 latent_size = latent_size,
                 num_heads = num_heads,
                 dropout = dropout,
                 device = device):
           super(EncoderBlock,self).__init__()

           self.latent_size = latent_size
           self.num_heads = num_heads
           self.dropout = dropout
           self.device = device

           # Normalization Layer
           self.Normalization = nn.LayerNorm(self.latent_size)

           # Multi-Headed Attention
           self.MultiHead = nn.MultiheadAttention(
                 embed_dim = self.latent_size,
                 num_heads = self.num_heads,
                 dropout = self.dropout
           )

           # MLP Layer
           self.encoder_MLP = nn.Sequential(
                 nn.Linear(self.latent_size,self.latent_size*4),
                 nn.GELU(),
                 nn.Dropout(self.dropout),
                 nn.Linear(self.latent_size*4,self.latent_size),
                 nn.Dropout(self.dropout)
           )


    def forward(self,embed_patches):
          
          # First Normalization Layer output
          first_normout = self.Normalization(embed_patches)

          # Output from the MultiHeadedAttention  
          attention_out = self.MultiHead(first_normout,first_normout,first_normout)[0]

          # First Residual Connection
          residual_out = attention_out + embed_patches

          # Second Norm output  
          second_normout = self.Normalization(residual_out)

          # Output from MLP
          mlp_out = self.encoder_MLP(second_normout)
          
          # Second residual connection
          output = mlp_out + residual_out
          
          return output

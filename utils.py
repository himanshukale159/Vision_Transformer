import torch

def get_config(cfg):

    patch_size = cfg['patch_size']
    latent_size = cfg['latent_size']
    n_channels = cfg['n_channels']
    num_heads = cfg['num_heads']
    num_encoders = cfg['num_encoders']
    dropout = cfg['dropout']
    num_classes = cfg['num_classes']
    size = cfg['size']
    batch_size = cfg['batch_size']

    return patch_size,latent_size,n_channels,num_heads,num_encoders,num_classes,dropout,size,batch_size
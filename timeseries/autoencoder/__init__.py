import os
import torch

from .model import *
from .. import utils

def autoencoder(input):
    path = utils.get_path('models', filename='autoencoder.pth')

    if not os.path.isfile(path):
        return None

    model = Autoencoder(seq_len=240, num_feats=2, latent_seq_len=1, latent_num_feats=8, hidden_size=128, num_layers=2)
    
    model.load_state_dict(torch.load(path))
    model.eval()

    return model(input)

__all__ = ['autoencoder']
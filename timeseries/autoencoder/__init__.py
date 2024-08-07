import torch
from .model import *
from .. import utils

path = utils.get_path('model', filename='autoencoder.pth')

def autoencoder(path):
    model = Autoencoder()
    
    model.load_state_dict(torch.load(path))
    model.eval()

    return model
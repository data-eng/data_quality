from . import eeg_features
from .eeg_datasets import create_spectrograms

def features(npz_object):
    return eeg_features.features(npz_object)
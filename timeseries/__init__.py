from . import eeg_features

def features(npz_object):
    return eeg_features.features(npz_object)
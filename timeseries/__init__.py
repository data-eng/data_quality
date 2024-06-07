from . import eeg_features

def features(npz_object):
    return eeg_features.features(npz_object)

def labels(npz_object):
    return npz_object["y"][:,3]

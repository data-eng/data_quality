import numpy
from . import eeg_features

def features(npz_object):
    return eeg_features.features(npz_object)

def labels(npz_object):
    numerical_labels = npz_object["y"][:,3]
    return numpy.array(list(map(str, numerical_labels)))

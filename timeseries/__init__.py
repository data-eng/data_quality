import numpy
from . import eeg_features
from . import task

def features(npz_object):
    return eeg_features.features(npz_object)

def labels(npz_object):
    numerical_labels = npz_object["y"][:,3]
    return numpy.array(list(map(str, numerical_labels)))

def create():
    return task.create()
def fit(model, X, y):
    return task.fit(model, X, y)
def score(model, X, y):
    return task.score(model, X, y)
def predict(model, X):
    return task.predict(model, X)

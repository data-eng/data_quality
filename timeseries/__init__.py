import numpy
from . import eeg_features
from . import mltask

def features(npz_object):
    return eeg_features.features(npz_object)

def labels(npz_object):
    numerical_labels = npz_object["y"][:,3]
    return numpy.array(list(map(str, numerical_labels)))

def task_create(X,y):
    m = mltask.newmodel()
    mltask.fit(m, X, y)
    task = { "model": m, "train_inputs": X, "train_outputs": y }
    return task

def task_evaluate(task, X, y):
    return mltask.score(task["model"], X, y)

def task_predict(task, X):
    return mltask.predict(task["model"], X)

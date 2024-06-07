import sys, glob
import numpy
import pandas
import json
import logging

import annotator_agreement
import timeseries

import sklearn.tree

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def make_stats(scores):
    """
    Compute statistics for the annotation agreement.

    :param scores: numpy array (n x 1) with agreement scores for each sample
    :return: tuple containing counts of good, mid, and bad agreements
    """
    good = 0
    mid = 0
    bad = 0

    for s in scores:
        if s > 0.75: good += 1
        elif s < 0.25: bad += 1
        else: mid += 1

    assert len(scores) == good + mid + bad

    with open('quality_stats.json', 'w') as json_file:
        json.dump({"Good": good, "Mid": mid, "Bad": bad}, json_file)

def loader( filenamelist ):
    features = None
    featurenames = None
    qfeat = None
    cfeat = None
    quality = None
    label = None
    for filename in filenamelist:
        npz_object = numpy.load(filename)
        (feat1,featnames1,qfeat1,cfeat1) = timeseries.features(npz_object)
        if features is None: features = feat1
        else: features = numpy.append(features, feat1, axis=0)
        if featurenames is None: featurenames = featnames1
        else: assert featurenames == featnames1
        if qfeat is None: qfeat = qfeat1
        else: assert qfeat == qfeat1
        if cfeat is None: cfeat = cfeat1
        else: assert cfeat == cfeat1
        
        quality1 = annotator_agreement.quality(npz_object)
        if quality is None: quality = quality1
        else: quality = numpy.append(quality, quality1, axis=0)

        label1 = timeseries.labels(npz_object)
        if label is None: label = label1
        else: label = numpy.append(label, label1, axis=0)

    assert quality.shape[0] == features.shape[0]
    assert quality.shape[0] == label.shape[0]
    return (features,quality,label,featurenames,qfeat,cfeat)


def main():
    """
    Process data from a .npz file, extract features, and compute quality metrics.
    
    :return: None
    """

    (features,quality,_,featurenames,qfeat,cfeat) = loader( glob.glob(f"data/*{sys.argv[1]}*npz") )

    logger.info("Using %s (%d samples) for training the quality model",
                sys.argv[1], quality.shape[0])

    qual_clf = sklearn.tree.DecisionTreeClassifier( max_depth=4 )
    qual_clf.fit( features[:,qfeat], list(map(str, quality)) )

    (features,_,labels,featurenames,qfeat,cfeat) = loader( glob.glob(f"data/*{sys.argv[2]}*npz") )
    
    logger.info("Using %s (%d samples) for training a classifier",
                sys.argv[2], features.shape[0])
    clf1 = sklearn.tree.DecisionTreeClassifier( max_depth=4 )
    clf1.fit( features[:,cfeat], list(map(str, labels)) )

    q = qual_clf.predict(features[:,qfeat])
    idx = numpy.where(q=="1.0")[0] # not sure why numpy.where returns a tuple
    logger.info("Among these, %d samples are high-quality. Using them to train another classifier.", idx.shape[0])
    clf2 = sklearn.tree.DecisionTreeClassifier( max_depth=4 )
    x = features[idx][:,cfeat]
    y = numpy.array(list(map(str, labels)))[idx]
    clf2.fit(x, y)

    (features,_,labels,featurenames,qfeat,cfeat) = loader( glob.glob(f"data/*{sys.argv[3]}*npz") )
    logger.info("Using %s (%d samples) for testing both classifiers",
                sys.argv[3], features.shape[0])
    y = numpy.array(list(map(str, labels)))
    logger.info("First: %f, second: %f",
                clf1.score(features[:,cfeat],y),
                clf2.score(features[:,cfeat],y))
    
    q = qual_clf.predict(features[:,qfeat])
    idx = numpy.where(q=="1.0")[0]
    x = features[idx][:,cfeat]
    y1 = y[idx]
    logger.info("Testing %d high-qual samples only. First: %f, second: %f",
                idx.shape[0], clf1.score(x,y1), clf2.score(x,y1))
    idx = numpy.where(q!="1.0")[0]
    x = features[idx][:,cfeat]
    y2 = y[idx]
    logger.info("Testing %d low-qual samples only. First: %f, second: %f",
                idx.shape[0], clf1.score(x,y2), clf2.score(x,y2))
    

import sys, glob
import numpy
import json
import logging

import annotator_agreement
import timeseries

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def loader( nameglob ):
    features = None
    featurenames = None
    qfeat = None
    cfeat = None
    quality = None
    label = None
    
    filenamelist = glob.glob( nameglob )
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
    #end for

    assert quality.shape[0] == features.shape[0]
    assert quality.shape[0] == label.shape[0]
    return (features,quality,label,featurenames,qfeat,cfeat)
#end def loader()


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
#end def make_stats()


def main():
    """
    Go through the whole process
    :return: None
    """

    (features,quality,_,featurenames,qfeat,cfeat) = loader( f"data/*{sys.argv[1]}*npz" )

    logger.info("Using %s (%d samples) for training the quality model",
                sys.argv[1], quality.shape[0])

    qual_clf = sklearn.tree.DecisionTreeRegressor( max_depth=6 )
    qual_clf.fit( features[:,qfeat], quality )

    (features,quality,labels,featurenames,qfeat,cfeat) = loader( f"data/*{sys.argv[2]}*npz" )
    logger.info("Testing on %s (%d samples) the quality model scored %f",
                sys.argv[2], features.shape[0],
                qual_clf.score(features[:,qfeat],quality))
    
    logger.info("Using %s (%d samples) for training a classifier",
                sys.argv[2], features.shape[0])
    model1 = timeseries.create()
    model1 = timeseries.fit(model1, features[:,cfeat], labels)

    q = qual_clf.predict(features[:,qfeat])
    idx = numpy.where(q>=0.8)[0] # numpy.where returns a tuple of arrays, one per dimension
    logger.info("%d samples are detected as high-quality. Using them to train a second classifier.", idx.shape[0])
    model2 = timeseries.create()
    x = features[idx][:,cfeat]
    y = labels[idx]
    model2 = timeseries.fit(model2, x, y)

    model3 = timeseries.create()
    idx = numpy.where(quality>=0.8)[0]
    x = features[idx][:,cfeat]
    y = labels[idx]
    logger.info("%d samples are really high-quality. Using them to train a third classifier.", idx.shape[0])
    model3 = timeseries.fit(model3, x, y)

    (features,_,labels,featurenames,qfeat,cfeat) = loader( f"data/*{sys.argv[3]}*npz" )
    logger.info("Using %s (%d samples) for testing all classifiers",
                sys.argv[3], features.shape[0])
    logger.info("First: %f, second: %f, third: %f",
                timeseries.score(model1, features[:,cfeat],labels),
                timeseries.score(model2, features[:,cfeat],labels),
                timeseries.score(model3, features[:,cfeat],labels))
    
    q = qual_clf.predict(features[:,qfeat])
    idx = numpy.where(q>=0.8)[0]
    x = features[idx][:,cfeat]
    y = labels[idx]
    logger.info("Testing %d high-qual samples only. First: %f, second: %f, third: %f",
                idx.shape[0],
                clf1.score(x,y), clf2.score(x,y), clf3.score(x,y))
    idx = numpy.where(q<0.8)[0]
    x = features[idx][:,cfeat]
    y = labels[idx]
    logger.info("Testing %d low-qual samples only. First: %f, second: %f, third: %f",
                idx.shape[0],
                timeseries.score(model1,x,y),
                timeseries.score(model2,x,y),
                timeseries.score(model3,x,y))
    

main()

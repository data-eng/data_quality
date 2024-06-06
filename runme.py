import sys
import numpy
import pandas
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

def main():
    """
    Process data from a .npz file, extract features, and compute quality metrics.
    
    :return: None
    """
    filename = sys.argv[1]
    npz_object = numpy.load(filename)

    features, fnames = timeseries.features(npz_object)
    quality = annotator_agreement.quality(npz_object)

    assert quality.shape[0] == features.shape[0]
 
    logger.info("Quality shape: %s", quality.shape)
    logger.info("Features shape: %s", features.shape)

    make_stats(scores=quality)

    df = pandas.DataFrame(features, columns=fnames)
    df["Quality"] = quality
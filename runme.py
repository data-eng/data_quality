import sys
import numpy
import pandas

import annotator_agreement
import timeseries


def make_stats( y ):
    good = 0
    mid = 0
    bad = 0
    count = 0
    # shape of y is (n,4), where each row has the annotation
    # from three annotators and the final decision.
    for i in range( y.shape[0] ):
        labels = y[i][0:3]
        ground_truth_label = y[i][3]
        a = inter_annotator_agreement( labels, ground_truth_label )
        if a > 0.75: good += 1
        elif a < 0.25: bad += 1
        else: mid += 1
    assert y.shape[0] == good + mid + bad
    return good, mid, bad


filename = sys.argv[1]
npz_object = numpy.load( filename )

features, fnames = timeseries.features( npz_object )
quality = annotator_agreement.quality( npz_object )

assert quality.shape[0] == features.shape[0]

print(quality.shape)
print(features.shape)

df = pandas.DataFrame( features, columns=fnames )
df["Quality"] = quality



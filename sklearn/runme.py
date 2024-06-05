import sys
import pandas
import numpy

import sklearn.tree
import sklearn.model_selection
#m sklearn.model_selection import KFold, cross_val_score

fname = sys.argv[1]
dfTrain = pandas.read_csv( fname )
y = list( map(str, dfTrain.pop("Quality").values) )

clf = sklearn.tree.DecisionTreeClassifier( max_depth=4 )
clf.fit( dfTrain, y )

fname = sys.argv[2]
dfTest = pandas.read_csv( fname )
y = numpy.array( list(map(str, dfTest.pop("Quality").values)) )

yy = clf.predict( dfTest )

print( "{} / {}".format(numpy.sum(y == yy),len(y)) )


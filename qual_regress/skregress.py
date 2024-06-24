import sklearn.tree

def newestimator():
    clf = sklearn.tree.DecisionTreeRegressor( max_depth=6 )
    return clf

def fit(clf, X, y):
    return clf.fit(X, y)

def score(clf, X, y):
    return clf.score(X, y)

def estimate(clf, X):
    return clf.predict(X)

import sklearn.tree

def create():
    clf = sklearn.tree.DecisionTreeClassifier( max_depth=6 )
    return clf

def fit(clf, X, y):
    return clf.fit(X, y)

def score(clf, X, y):
    return clf.fit(X, y)

def predict(clf, X, y):
    return clf.predict(X)

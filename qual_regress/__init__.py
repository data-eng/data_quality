from . import skregress

def estimator_create(X,y):
    m = skregress.newestimator()
    skregress.fit(m, X, y)
    task = { "model": m, "train_inputs": X, "train_outputs": y }
    return task

def estimator_evaluate(task, X, y):
    return skregress.score(task["model"], X, y)

def estimator_apply(task, X):
    return skregress.estimate(task["model"], X)

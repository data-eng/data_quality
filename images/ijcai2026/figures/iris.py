import pandas
import numpy as np

import sklearn
import matplotlib.pyplot as plt


def select( z, n ):
    zz = z.reshape( (1,z.shape[0]) )
    dists = abs(zz.T - zz)
    start = int(np.argmin(dists.mean(axis=1)))
    chosen = [start]
    min_d = dists[start]
    while len(chosen) < n:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        new_d = abs(z-z[nxt])
        min_d = np.minimum(min_d, new_d)
    return np.array(chosen, dtype=int)


#labels = iris["class"]
#setosa = (labels=="Iris-setosa")
#Ec = features[setosa]


iris = sklearn.datasets.load_iris()

#sts = sklearn.preprocessing.normalize( iris.data[iris.target==0], axis=0, copy=True )
#vrs = sklearn.preprocessing.normalize( iris.data[iris.target==1], axis=0, copy=True ) 
#vgn = sklearn.preprocessing.normalize( iris.data[iris.target==2], axis=0, copy=True ) 
sts = iris.data[iris.target==0]
vrs = iris.data[iris.target==1]
vgn = iris.data[iris.target==2]

sts = np.array( [complex(sts[i,0],sts[i,1]) for i in range(sts.shape[0])] )
sel0 = select( sts, 16 )
vrs = np.array( [complex(vrs[i,0],vrs[i,1]) for i in range(vrs.shape[0])] )
sel1 = select( vrs, 16 )

_, ax = plt.subplots()
xx = iris.data[0:100,0]
yy = iris.data[0:100,1]
scatter = ax.scatter( xx, yy, c=iris.target[0:100], cmap="jet" )
for order,idx in enumerate(sel0):
    ax.annotate( order, (xx[idx],yy[idx]), color="black" )
for order,idx in enumerate(sel1):
    ax.annotate( order, (xx[50+idx],yy[50+idx]), color="black" )
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)




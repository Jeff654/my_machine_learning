# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# build a classification task using 3 informative features
x, y = make_classification(n_samples = 1000, n_features = 25, n_informative = 3, n_redundant = 2, n_repeated = 0, n_classes = 8, n_clusters_per_class = 1, random_state = 0)

# create the RFE object and compute a cross-validation score
svc = SVC(kernel = 'linear')
rfecv = RFECV(estimator = svc, step = 1, cv = StratifiedKFold(y, 2), scoring = 'accuracy')
rfecv.fit(x, y)

print("optimal number of features: %d" % rfecv.n_features_)

plt.figure()
plt.xlabel('number of features selected')
plt.ylabel('cross validation score (nb of correct classification)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

boston = load_boston()
x, y = boston['data'], boston['target']

# use the base estimatorLassoCV since the L1 norm promotes sparsity of features
clf = LassoCV()

# set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold = 0.25)
sfm.fit(x, y)
n_features = sfm.transform(x).shape[1]

# reset the threshold till the number of features equals two, the attribute can be set directly instead of repeatedly fitting the metatransformer
while n_features > 2:
	sfm.threshold += 0.1
	x_transform = sfm.transform(x)
	n_features = x_transform.shape[1]

plt.title("features selected from Boston using SelectFromModel with threshold %0.3f." % sfm.threshold)

feature1 = x_transform[:, 0]
feature2 = x_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("feature number 1")
plt.ylabel("feature number 2")
plt.xlim([np.min(feature2), np.max(feature2)])
plt.show()


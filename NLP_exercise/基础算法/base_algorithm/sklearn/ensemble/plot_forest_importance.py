# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

x, y = make_classification(n_samples = 1000, n_features = 10, n_informative = 3, n_redundant = 0, n_classes = 2, random_state = 0, shuffle = False)

forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
forest.fit(x, y)
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]
print("Feature ranking")

for f in range(x.shape[1]):
	print("%d, feature %d (%f)" %(f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("feature importances")
plt.bar(range(x.shape[1]), importances[indices], color = 'r', yerr = std[indices], align = 'center')
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()




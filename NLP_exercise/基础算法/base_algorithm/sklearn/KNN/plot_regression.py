# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
x = np.sort(5 * np.random.rand(40, 1), axis = 0)
t = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(x).ravel()

# add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

n_neighbors = 5
for i, weights in enumerate(['uniform', 'distance']):
	knn = neighbors.KNeighborsRegressor(n_neighbors, weights = weights)
	y_ = knn.fit(x, y).predict(t)

	plt.subplot(2, 1, i + 1)
	plt.scatter(x, y, c = 'k', label = 'data')
	plt.plot(t, y_, c = 'g', label = 'prediction')
	plt.axis('tight')
	plt.legend()
	plt.title("kNeighborsRegression (k = %i, weights = '%s')" %(n_neighbors, weights))

plt.show()



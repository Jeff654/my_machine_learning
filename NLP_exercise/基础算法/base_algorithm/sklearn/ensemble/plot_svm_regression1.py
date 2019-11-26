# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# generate sample data
x = np.sort(5 * np.random.rand(40, 1), axis = 0)
# y = np.sin(x).ravel()

y = (x * x).ravel()
# add noise to targets
y[::5] += (0.5 - np.random.rand(8))

colors = 'rgbyk'
gammas = [0.01, 0.04, 0.08, 0.12, 0.16]
r = -1

for gamma, color in zip(gammas, colors):
	my_estimator = SVR(kernel = 'sigmoid', gamma = gamma, coef0 = r)
	y_predict = my_estimator.fit(x, y).predict(x)

	plt.plot(x, y_predict, c = color, label = 'gamma = %.3f' %gamma)

"""
for gamma in gammas:
	my_estimator = SVR(kernel = 'sigmoid', gamma = gamma, coef0 = r)
	y_predict = my_estimator.fit(x, y).predict(x)
	

	plt.hold('on')
	plt.plot(x, y_predict, c = 'r', label = 'gamma = %.3f' %gamma)
"""


plt.scatter(x, y, c = 'k', label = 'data')

plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()


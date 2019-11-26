# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# generate sample data
x = np.sort(5 * np.random.rand(40, 1), axis = 0)
# y = np.sin(x).ravel()

y = x.ravel()
# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

gammas = [0.01, 0.03, 0.06, 0.08, 0.11, 0.13, 0.16]
r = -1

for gamma in gammas:
	my_estimator = SVR(kernel = 'sigmoid', C = 1e3, gamma = gamma, coef0 = r)
	y_predict = my_estimator.fit(x, y).predict(x)
	
	plt.hold('on')
	plt.plot(x, y_predict, c = 'r', label = 'gamma = %.3f' %gamma)

plt.scatter(x, y, c = 'k', label = 'data')

plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()



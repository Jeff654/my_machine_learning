# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(1)
x = np.sort(200 * rng.rand(100, 1) - 100, axis = 0)
y = np.array([np.pi * np.sin(x).ravel(), np.pi * np.cos(x).ravel()]).T
y[::5, :] += (0.5 - rng.rand(20, 2))

regressor_1 = DecisionTreeRegressor(max_depth = 2)
regressor_2 = DecisionTreeRegressor(max_depth = 5)
regressor_3 = DecisionTreeRegressor(max_depth = 8)
regressor_1.fit(x, y)
regressor_2.fit(x, y)
regressor_3.fit(x, y)

x_test = np.arange(-100, 100, 0.01)[:, np.newaxis]
y_1 = regressor_1.predict(x_test)
y_2 = regressor_2.predict(x_test)
y_3 = regressor_3.predict(x_test)

plt.figure()
plt.scatter(y[:, 0], y[:, 1], c = 'k', label = 'data')
plt.scatter(y_1[:, 0], y_1[:, 1], c = 'g', label = 'max_depth = 2')
plt.scatter(y_2[:, 0], y_2[:, 1], c = 'r', label = 'max_depth = 5')
plt.scatter(y_3[:, 0], y_3[:, 1], c = 'b', label = 'max_depth = 8')

plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("data")
plt.ylabel("target")
plt.title("multi-output decision tree regression")
plt.legend()
plt.show()



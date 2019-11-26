# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(1)
x = np.sort(5 * rng.rand(80, 1), axis = 0)
y = np.sin(x).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

regressor_1 = DecisionTreeRegressor(max_depth = 2)
regressor_2 = DecisionTreeRegressor(max_depth = 5)
regressor_1.fit(x, y)
regressor_2.fit(x, y)

x_test = np.arange(0, 5.0, 0.01)[:, np.newaxis]
y_1 = regressor_1.predict(x_test)
y_2 = regressor_2.predict(x_test)


plt.figure()
plt.scatter(x, y, c = 'k', label = 'data')
plt.plot(x_test, y_1, c = 'b', label = 'max_depth = 2', linewidth = 2)
plt.plot(x_test, y_2, c = 'r', label = 'max_depth = 5', linewidth = 5)
plt.xlabel('data')
plt.ylabel('target')
plt.title('decision tree regression')
plt.legend()
plt.show()



# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# generate sample data
x = 0.3 * np.random.randn(100, 2)
x_train = np.r_[x + 2, x - 2]

# generate some regular novel observetions
x = 0.3 * np.random.randn(20, 2)
x_test = np.r_[x + 2, x - 2]

# generate some abnormal novel observations
x_outliers = np.random.uniform(low = -4, high = 4, size = (20, 2))


# fit the model
clf = svm.OneClassSVM(nu = 0.1, kernel = 'rbf', gamma = 0.1)
clf.fit(x_train)

y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)
y_pred_outliers = clf.predict(x_outliers)

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == -1].size


# plot the line, the points, and the nearest vectors to the plane
z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.title('Novelty Detection')
plt.contourf(xx, yy, z, levels = np.linspace(z.min(), 0, 7), cmap = plt.cm.Blues_r)
a = plt.contour(xx, yy, z, levels = [0], linewidths = 2, colors = 'red')
plt.contourf(xx, yy, z, levels = [0, z.max()], colors = 'orange')

b1 = plt.scatter(x_train[:, 0], x_train[:, 1], c = 'white')
b2 = plt.scatter(x_test[:, 0], x_test[:, 1], c = 'green')
c = plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c = 'red')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

plt.legend([a.collections[0], b1, b2, c], ['learned frontier', 'training observations'], loc = 'upper left', prop = matplotlib.font_manager.FontProperties(size = 11))
plt.xlabel("error train: %d / 200; errors novel regular: %d / 40; errors novel abnormal: %d / 40" %(n_error_train, n_error_test, n_error_outliers))
plt.show()



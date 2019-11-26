# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
np.random.seed(0)

# generate xor data
x = np.random.randn(300, 2)
y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)

clf = svm.NuSVC()
clf.fit(x, y)

# plot the decision function for each datapoint on the grid
z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.imshow(z, interpolation = 'nearest', aspect = 'auto', extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin = 'lower', cmap = cm.RdYlGn)

contours = plt.contour(xx, yy, z, levels = [0], linewidths = 2, linetypes=  '--')
plt.scatter(x[:, 0], x[:, 1], s = 30, c = y, cmap = plt.cm.Paired)
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()




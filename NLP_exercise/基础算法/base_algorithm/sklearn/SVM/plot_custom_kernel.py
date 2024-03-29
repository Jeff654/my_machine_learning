# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()

# only take the first two features
x = iris.data[:, :2]
y = iris.target

def my_kernel(x, y):
	M = np.array([[2, 0], [0, 1.0]])
	return np.dot(np.dot(x, M), y.T)

# step size in the mesh
h = 0.02

# create on the instance of SVM and fit data
clf = svm.SVC(kernel = my_kernel)
clf.fit(x, y)

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.pcolormesh(xx, yy, z, cmap = plt.cm.Paired)
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Paired)
plt.title('3-class classification using SVM with custom kernel')
plt.axis('tight')
plt.show()

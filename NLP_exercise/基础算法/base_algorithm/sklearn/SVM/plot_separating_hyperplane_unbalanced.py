# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import SGDClassifier


# create 40 separate points
rng = np.random.RandomState(0)

n_samples1 = 1000
n_samples2 = 100

x = np.r_[1.5 * rng.randn(n_samples1, 2), 0.5 * rng.randn(n_samples2, 2) + [2, 2]]

y = [0] * n_samples1 + [1] * n_samples2

# fit the model and get the separating hyperplane
# clf = svm.SVC(kernel = 'linear', C = 1.0)
clf = SGDClassifier(n_iter = 100, alpha = 0.01)
clf.fit(x, y)

w = clf.coef_[0]
print 'w: ', w

a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]


# get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel = 'linear', class_weight = {1: 10})
wclf.fit(x, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]


# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label = 'no weights')
h1 = plt.plot(xx, wyy, 'k--', label = 'with weights')
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Paired)
plt.legend()

plt.axis('tight')
plt.show()




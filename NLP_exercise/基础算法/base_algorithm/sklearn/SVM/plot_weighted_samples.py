# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def plot_decision_function(classifier, sample_weight, axis, title):
	# plot decision function
	xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
	z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)

	# plot the line, the points, and the nearest vectors to the plane
	axis.contourf(xx, yy, z, alpha = 0.75, cmap = plt.cm.bone)
	axis.scatter(x[:, 0], x[:, 1], c = y, s = 100 * sample_weight, alpha = 0.9, cmap = plt.cm.bone)

	axis.axis('off')
	axis.set_title(title)


# create 20 points
np.random.seed(0)
x = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
y = [1] * 10 + [-1] * 10

sample_weight_last_ten = abs(np.random.randn(len(x)))
sample_weight_constant = np.ones(len(x))

# let bigger weights some outliers
sample_weight_last_ten[15: ] *= 5
sample_weight_constant[9] *= 15

# fit the model
clf_weights = svm.SVC()
clf_weights.fit(x, y, sample_weight = sample_weight_last_ten)

clf_no_weights = svm.SVC()
clf_no_weights.fit(x, y)

fig, axes = plt.subplots(1, 2, figsize = (14, 6))
plot_decision_function(clf_no_weights, sample_weight_constant, axes[0], "constant weights")
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1], 'modified weights')

plt.show()







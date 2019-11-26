# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
n_estimators = 30
plot_colors = 'ryb'
cmap = plt.cm.RdYlBu

# fine step width for decision surface contours
plot_step = 0.02

# step width for coarse classifier guesses
plot_step_coarser = 0.5
RANDOM_SEED = 13

iris = load_iris()
plot_idx = 1

models = [DecisionTreeClassifier(max_depth = None), RandomForestClassifier(n_estimators = n_estimators), ExtraTreesClassifier(n_estimators = n_estimators), AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3), n_estimators = n_estimators)]

# 每次选取两维特征
for pair in ([0, 1], [0, 2], [2, 3]):
	for model in models:
		x = iris.data[:, pair]
		y = iris.target

		# shuffle
		idx = np.arange(x.shape[0])
		np.random.seed(RANDOM_SEED)
		np.random.shuffle(idx)
		x = x[idx]
		y = y[idx]

		# standardize
		mean = x.mean(axis = 0)
		std = x.std(axis = 0)
		x = (x - mean) / std


		clf = clone(model)
		clf = model.fit(x, y)

		scores = clf.score(x, y)
		model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
		model_details = model_title
		if hasattr(model, "estimators_"):
			model_details += " with {} estimators".format(len(model.estimators_))

		print( model_details + " with features", pair, "has a score of", scores)

		plt.subplot(3, 4, plot_idx)
		if plot_idx <= len(models):
			# add title at the top of each column
			plt.title(model_title)

		x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
		y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
		
		xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
		

		# plot either a single DecisionTreeClassifier or alpha blend the decision surfaces of the ensemble of calssifiers
		if isinstance(model, DecisionTreeClassifier):
			z = model.predict(np.c_[xx.ravel(), yy.ravel()])
			z = z.reshape(xx.shape)
			cs = plt.contourf(xx, yy, z, cmap = cmap)
		else:
			# choose alpha blend level with respect to the number of estimators that are in use
			estimator_alpha = 1.0 / len(model.estimators_)
			for tree in model.estimators_:
				z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
				z = z.reshape(xx.shape)
				cs = plt.contourf(xx, yy, z, alpha = estimator_alpha, cmap = cmap)

		
		xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser), np.arange(y_min, y_max, plot_step_coarser))
		z_points_coarse = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
		cs_points = plt.scatter(xx_coarser, yy_coarser, s = 15, c = z_points_coarse, cmap = cmap, edgecolor = "none")


		# plot the training data, these are clustered together and have a black outline
		for i, c in zip(xrange(n_classes), plot_colors):
			idx = np.where(y == i)
			plt.scatter(x[idx, 0], x[idx, 1], c = c, label = iris.target_names[i], cmap = cmap)

		plot_idx += 1

plt.suptitle('classifiers on feature subsets of the iris dataset')
plt.axis('tight')
plt.show()



# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# number of iterations for computing expections
n_repeat = 50
n_train = 50
n_test = 1000
noise = 0.1
np.random.seed(0)

# change this for exploring the bias-variance decomposition of other estimators, this should work well for estimators with high variance(eg..decision tree or KNN), but poorly for estimators with low variance(eg..linear models)

estimators = [("Tree", DecisionTreeRegressor()), ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]
n_estimators = len(estimators)

# generate the data
def f(x):
	x = x.ravel()
	return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise, n_repeat = 1):
	x = np.random.rand(n_samples) * 10 - 5
	x = np.sort(x)

	if n_repeat == 1:
		y = f(x) + np.random.normal(0.0, noise, n_samples)
	else:
		y = np.zeros((n_samples, n_repeat))
		for i in range(n_repeat):
			y[:, 1] = f(x) + np.random.normal(0.0, noise, n_samples)
	x = x.reshape((n_samples, 1))
	return x, y

x_train = []
y_train = []

for i in range(n_repeat):
	x, y = generate(n_samples = n_train, noise = noise)
	x_train.append(x)
	y_train.append(y)

x_test, y_test = generate(n_samples = n_test, noise = noise, n_repeat = n_repeat)

for n, (name, estimator) in enumerate(estimators):
	y_predict = np.zeros((n_test, n_repeat))

	for i in range(n_repeat):
		estimator.fit(x_train[i], y_train[i])
		y_predict[:, i] = estimator.predict(x_test)
	
	# bias^2 + variance + noise decomposition of the mean squared error
	y_error = np.zeros(n_test)

	for i in range(n_repeat):
		for j in range(n_repeat):
			y_error += (y_test[:, j] - y_predict[:, i]) ** 2
	
	y_error /= (n_repeat * n_repeat)

	y_noise = np.var(y_test, axis = 1)
	y_bias = (f(x_test) - np.mean(y_predict, axis = 1)) ** 2
	y_var = np.var(y_predict, axis = 1)

	print("{0}: {1: .4f} (error) = {2: .4f} (bias^2) + {3: .4f} (var) + {4: .4f} (noise)".format(name, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise)))

	# plot figures
	plt.subplot(2, n_estimators, n + 1)
	plt.plot(x_test, f(x_test), 'b', label = "$f(x)$")
	plt.plot(x_train[0], y_train[0], '.b', label = "LS ~ $ y = f(x) + noise $")

	for i in range(n_repeat):
		if i == 0:
			plt.plot(x_test, y_predict[:, i], 'r', label = "$\^y(x)$")
		else:
			plt.plot(x_test, y_predict[:, i], 'r', alpha = 0.05)
	
	plt.plot(x_test, np.mean(y_predict, axis = 1), 'c', label = "$\mathbb{E}_{LS} \^y(x)$")
	plt.xlim([-5, 5])
	plt.title(name)


	if n == 0:
		plt.legend(loc = 'upper left', prop = {"size": 11})
	
	plt.subplot(2, n_estimators, n_estimators + n + 1)
	plt.plot(x_test, y_error, "r", label="$error(x)$")
	plt.plot(x_test, y_bias, "b", label="$bias^2(x)$"),
	plt.plot(x_test, y_var, "g", label="$variance(x)$"),
	plt.plot(x_test, y_noise, "c", label="$noise(x)$")

	plt.xlim([-5, 5])
	plt.ylim([0, 0.1])
	
	if n == 0:
		plt.legend(loc = 'upper left', prop = {'size': 11})

plt.show()




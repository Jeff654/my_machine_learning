# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression, ARDRegression


# generated simulated data with Gaussian weights
np.random.seed(0)
n_samples, n_features = 100, 100

# generate Gaussion data
x = np.random.randn(n_samples, n_features)

# create weights with a precision lambda_ of 4
lambda_ = 4.0
w = np.zeros(n_features)

# only keep 10 weights of interest
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
	w[i] = stats.norm.rvs(loc = 0, scale = 1.0 / np.sqrt(lambda_))


# create noise with a precision alpha_ of 50
alpha_ = 50
noise = stats.norm.rvs(loc = 0, scale = 1.0 / np.sqrt(alpha_), size = n_samples)

# create the target
y = np.dot(x, w) + noise

######################################################################
# fit the Bayesian Ridge Regression, Automatic Relevance Determination Regression(ARD) and OLS for comparsion
clf = BayesianRidge(compute_score = True)
clf.fit(x, y)

ard = ARDRegression(compute_score = True)
ard.fit(x, y)

ols = LinearRegression()
ols.fit(x, y)

######################################################################
# plot true weights, estimatedweights and histogram of the weights
plt.figure(figsize = (6, 5))
plt.title("weights of the model")
plt.plot(clf.coef_, 'b-', label = 'Bayesian Ridge estimate')
plt.plot(w, 'g-', label = 'Ground truth')
plt.plot(ols.coef_, 'r--', label = 'OLS estimate')
plt.plot(ard.coef_, 'ko', label = 'ARDRegression')

plt.xlabel("features")
plt.ylabel("value of the weights")
plt.legend(loc = 'best', prop = dict(size = 12))


plt.figure(figsize = (6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins = n_features, log = True)
plt.plot(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)), 'ro', label = 'Relevant features')
plt.xlabel("values of weights")
plt.ylabel("weights")
plt.legend(loc = 'lower left')


plt.figure(figsize = (6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_, 'g-')
plt.plot(ard.scores_, 'b--')
plt.xlabel("Iterations")
plt.ylabel('Score')
plt.show()



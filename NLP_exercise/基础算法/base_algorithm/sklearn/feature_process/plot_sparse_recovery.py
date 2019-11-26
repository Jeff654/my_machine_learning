# -*- coding: utf-8 -*-

import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

from sklearn.linear_model import RandomizedLasso, lasso_stability_path, LassoLarsCV
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.utils import ConvergenceWarning


def mutual_incoherence(x_relevant, x_irelevant):
	projector = np.dot(np.dot(x_irelevant.T, x_relevant), pinvh(np.dot(x_relevant.T, x_relevant)))
	return np.max(np.abs(projector).sum(axis = 1))


for conditioning in (1, 1e-4):
	# simulate regression data with a correlated design
	n_features = 501
	n_relevant_features = 3
	noise_level = 0.2
	coef_min = 0.2

	# the Donoho-Tanner phase transition is around n_samples = 25: below we will completely fail to recover in the well-conditioned case
	n_samples = 25
	block_size = n_relevant_features
	rng = np.random.RandomState(42)

	# the coefficients of our model
	coef = np.zeros(n_features)
	coef[: n_relevant_features] = coef_min + rng.rand(n_relevant_features)

	# the correction of our design: variables corrected by blocs of 3
	corr = np.zeros((n_features, n_features))
	for i in range(0, n_features, block_size):
		corr[i: i + block_size, i: i + block_size] = 1 - conditioning
	corr.flat[::n_features + 1] = 1
	corr = linalg.cholesky(corr)

	# our design
	x = rng.normal(size = (n_samples, n_features))
	x = np.dot(x, corr)
	x[: n_relevant_features] /= np.abs(linalg.svdvals(x[: n_relevant_features])).max()
	x = StandardScaler().fit_transform(x.copy())

	# the output variable
	y = np.dot(x, coef)
	y /= np.std(y)
	y += noise_level * rng.normal(size = n_samples)
	mi = mutual_incoherence(x[:, :n_relevant_features], x[:, n_relevant_features:])


	# plot stability selection path, using a high eps for early stopping of the path, to save computing time
	alpha_grid, scores_path = lasso_stability_path(x, y, random_state = 42, eps = 0.05)

	plt.figure()
	hg = plt.plot(alpha_grid[1:]**0.33, scores_path[coef != 0].T[1:], 'r')
	hb = plt.plot(alpha_grid[1:] ** .333, scores_path[coef == 0].T[1:], 'k')
	ymin, ymax = plt.ylim()
	
	plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
	plt.ylabel('Stability score: proportion of times selected')
	plt.title('Stability Scores Path - Mutual incoherence: %.1f' % mi)
	plt.axis('tight')
	plt.legend((hg[0], hb[0]), ('relevant features', 'irrelevant features'), loc='best')


	###############################################################
	# Plot the estimated stability scores for a given alpha, Use 6-fold cross-validation rather than the default 3-fold: it leads to a better choice of alpha: Stop the user warnings outputs- they are not necessary for the example as it is specifically set up to be challenging.
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', UserWarning)
		warnings.simplefilter('ignore', ConvergenceWarning)
		lars_cv = LassoLarsCV(cv=6).fit(x, y)

	# Run the RandomizedLasso: we use a paths going down to .1*alpha_max to avoid exploring the regime in which very noisy variables enter the model
	alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
	clf = RandomizedLasso(alpha=alphas, random_state=42).fit(x, y)
	trees = ExtraTreesRegressor(100).fit(x, y)
	
	# Compare with F-score
	F, _ = f_regression(x, y)

	plt.figure()
	for name, score in [('F-test', F), 
			('Stability selection', clf.scores_), 
			('Lasso coefs', np.abs(lars_cv.coef_)), 
			('Trees', trees.feature_importances_)]:
		precision, recall, thresholds = precision_recall_curve(coef != 0, score)
		plt.semilogy(np.maximum(score / np.max(score), 1e-4), label="%s. AUC: %.3f" % (name, auc(recall, precision)))
	
	
	plt.plot(np.where(coef != 0)[0], [2e-4] * n_relevant_features, 'mo', label="Ground truth")
	plt.xlabel("Features")
	plt.ylabel("Score")
	
	# Plot only the 100 first coefficients
	plt.xlim(0, 100)
	plt.legend(loc='best')
	plt.title('Feature selection scores - Mutual incoherence: %.1f' % mi)
plt.show()


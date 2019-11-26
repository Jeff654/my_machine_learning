# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline


digits = datasets.load_digits()
y = digits.target

# throw away some data
y = y[:200]
x = digits.data[:200]
n_samples = len(y)
x = x.reshape((n_samples, -1))

# add 200 non-informative features
x = np.hstack((x, 2 * np.random.random((n_samples, 200))))

# create a feature-selection transform and an instance of SVM that we combine together to have an full-blown estimator
transform = feature_selection.SelectPercentile(feature_selection.f_classif)

clf = Pipeline([('anova', transform), ('svc', svm.SVC(C = 1.0))])


# plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
	clf.set_params(anova__percentile = percentile)

	# compute cross-validation score using all CPUs
	this_scores = cross_validation.cross_val_score(clf, x, y, n_jobs = 1)
	score_means.append(this_scores.mean())
	score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title('performance of the SVM-anova varying the percentile of feature selected')
plt.xlabel('percentile')
plt.ylabel('percention rate')

plt.axis('tight')
plt.show()





# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import MultiTaskLasso, Lasso, ElasticNet, MultiTaskElasticNet

rng = np.random.RandomState(42)

# generate some 2D coefficients with sine waves with random frequency and phase
n_samples, n_features, n_tasks = 100, 30, 40
n_relevant_features = 5

coef = np.zeros((n_tasks, n_features))
times = np.linspace(0, 2 * np.pi, n_tasks)

for k in range(n_relevant_features):
	coef[:, k] = np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))

x = rng.randn(n_samples, n_features)
y = np.dot(x, coef.T) + rng.randn(n_samples, n_tasks)

coef_lasso_ = np.array([Lasso(alpha = 0.5).fit(x, y).coef_ for y in y.T])
coef_multi_task_lasso_ = ElasticNet(alpha = 1.0).fit(x, y).coef_


###################################################################

# plot support and time series
fig = plt.figure(figsize = (8, 5))

plt.subplot(1, 2, 1)
plt.spy(coef_lasso_)
plt.xlabel('Feature')
plt.ylabel("Time (or task)")
plt.text(10, 5, 'Lasso')


plt.subplot(1, 2, 2)
plt.spy(coef_multi_task_lasso_)
plt.xlabel('Feature')
plt.ylabel('Time (or Task)')
plt.text(10, 5, 'MultiTaskLasso')

fig.suptitle('Coefficient non-zero location')


feature_to_plot = 0

plt.figure()
plt.plot(coef[:, feature_to_plot], 'k', label = 'Ground truth')
plt.plot(coef_lasso_[:, feature_to_plot], 'g', label = 'Lasso')
plt.plot(coef_multi_task_lasso_[:, feature_to_plot], 'r', label = 'MultiTaskLasso')

plt.legend(loc = 'upper center')
plt.axis('tight')
plt.ylim([-1.1, 1.1])
plt.show()











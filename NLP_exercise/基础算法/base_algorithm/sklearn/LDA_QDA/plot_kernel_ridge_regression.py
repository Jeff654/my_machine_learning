# -*- coding: utf-8 -*-


import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR


rng = np.random.seed(0)

# generate sample data
x = 5 * np.random.rand(10000, 1)
y = np.sin(x).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(x.shape[0] / 5))
x_plot = np.linspace(0, 5, 100000)[:, None]


##############################################################
# fit regression model
train_size = 100
svr = GridSearchCV(SVR(kernel = 'rbf', gamma = 0.1), cv = 5, 
		param_grid = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)})

kr = GridSearchCV(KernelRidge(kernel = 'rbf', gamma = 0.1), 
		param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})


t0 = time.time()
svr.fit(x[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)


t0 = time.time()
kr.fit(x[:train_size], y[:train_size])
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("Support vector ratio: %.3f" % sv_ratio)


t0 = time.time()
y_svr = svr.predict(x_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" %(x_plot.shape[0], svr_predict))


t0 = time.time()
y_kr = kr.predict(x_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" %(x_plot.shape[0], kr_predict))



########################################################################
# plot the results
# 返回支持向量的索引
sv_ind = svr.best_estimator_.support_

plt.scatter(x[sv_ind], y[sv_ind], c = 'r', label = 'SVR support vectors')
plt.scatter(x[:100], y[:100], c = 'k', label = 'data')
plt.hold('on')
plt.plot(x_plot, y_svr, c = 'r', label = 'SVR (fit: %.3fs, predict: %.3fs)' %(svr_fit, svr_predict))

plt.plot(x_plot, y_kr, c = 'g', label = 'KRR (fit: %.3fs, predict: %.3fs)' %(kr_fit, kr_predict))

plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()

plt.figure()


# generate sample data
x = 5 * np.random.rand(10000, 1)
y = np.sin(x).ravel()

y[::5] += 3 * (0.5 - np.random.rand(x.shape[0] / 5))
sizes = np.logspace(1, 4, 7)

for name, estimator in {'KRR': KernelRidge(kernel = 'rbf', alpha = 0.1, gamma = 10), 'SVR': SVR(kernel = 'rbf', C = 1e1, gamma = 10)}.items():
	train_time = []
	test_time = []

	for train_test_size in sizes:
		t0 = time.time()
		estimator.fit(x[:train_test_size], y[:train_test_size])
		train_time.append(time.time() - t0)

		t0 = time.time()
		estimator.predict(x_plot[:100])
		test_time.append(time.time() - t0)
	
	plt.plot(sizes, train_time, 'o-', color = 'r' if name == 'SVR' else 'g', label = "%s (train)" %name)
	plt.plot(sizes, test_time, 'o--', color = 'r' if name == 'SVR' else 'g', label = "%s (test)" %name)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("train size")
plt.ylabel("time (seconds)")
plt.title('execution time')
plt.legend(loc = 'best')


#####################################################################
# visualize the learning curve
plt.figure()

svr = SVR(kernel = 'rbf', C = 10, gamma = 0)
kr = KernelRidge(kernel = 'rbf', alpha = 0.1, gamma = 0.1)

train_sizes, train_scores_svr, test_scores_svr = learning_curve(svr, x[:100], y[:100], train_sizes = np.linspace(0.1, 1, 10), scoring = "mean_squared_error", cv = 10)

train_sizes_abs, train_scores_kr, test_scores_kr = learning_curve(kr, x[:100], y[:100], train_sizes = np.linspace(0.1, 1, 10), scoring = "mean_squared_error", cv = 10)

plt.plot(train_sizes, test_scores_svr.mean(1), 'o-', color = 'r', label = 'SVR')
plt.plot(train_sizes, test_scores_kr.mean(1), 'o-', color = 'g', label = 'KRR')

plt.xlabel("train size")
plt.ylabel("mean squared error")
plt.title("learning curves")
plt.legend(loc = 'best')
plt.show()



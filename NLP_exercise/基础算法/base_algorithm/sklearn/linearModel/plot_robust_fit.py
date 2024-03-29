# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


np.random.seed(42)

x = np.random.normal(size = 400)
y = np.sin(x)

# make sure that it x is 2D
x = x[:, np.newaxis]

x_test = np.random.normal(size = 200)
y_test = np.sin(x_test)
x_test = x_test[:, np.newaxis]

y_errors = y.copy()
y_errors[::3] = 3

x_errors = x.copy()
x_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

x_errors_large = x.copy()
x_errors_large[::3] = 10


estimators = [('OLS', linear_model.LinearRegression()), 
		('Theil-Sen', linear_model.TheilSenRegressor(random_state = 42)), 
		('RANSAC', linear_model.RANSACRegressor(random_state = 42)),]

x_plot = np.linspace(x.min(), x.max())

for title, this_x, this_y in [('Modeling errors only', x, y), 
				('Corrupt x, small deviants', x_errors, y), 
				('corrupt y, small deviants', x, y_errors), 
				('corrupt x, large deviants', x_errors_large, y), 
				('corrupt y, large deviants', x, y_errors_large)]:
	plt.figure(figsize = (5, 4))
	plt.plot(this_x[:, 0], this_y, 'k+')

	for name, estimator in estimators:
		model = make_pipeline(PolynomialFeatures(3), estimator)
		model.fit(this_x, this_y)

		mse = metrics.mean_squared_error(model.predict(x_test), y_test)
		y_plot = model.predict(x_plot[:, np.newaxis])
		plt.plot(x_plot, y_plot, label = '%s: error = %.3f' %(name, mse))
	
	plt.legend(loc = 'best', frameon = False, title = 'Error: mean absolute deviation \n to non corrupt data')
	
	plt.xlim(-4, 10.2)
	plt.ylim(-2, 10.2)
	plt.title(title)
plt.show()



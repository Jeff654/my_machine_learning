# -*- coding: utf-8 -*-


import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

n_samples = 1000
n_outliers = 50

# generate a random regression problem
x, y, coef = datasets.make_regression(n_samples = n_samples, n_features = 1, n_informative = 1, noise = 10, coef = True, random_state = 0)

# add outlier data
np.random.seed(0)
x[: n_outliers] = 3 + 0.5 * np.random.normal(size = (n_outliers, 1))
y[: n_outliers] = -3 + 10 * np.random.normal(size = n_outliers)

# fit the line using all data
model = linear_model.LinearRegression()
model.fit(x, y)

# robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(x, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# predict data of estimated coefficients
line_x = np.arange(-5, 5)
line_y = model.predict(line_x[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_x[:, np.newaxis])


# compare estimated coefficients
print("Estimated coefficients (true, normal, RANSAC): ")
print(coef, model.coef_, model_ransac.estimator_.coef_)


plt.plot(x[inlier_mask], y[inlier_mask], '.g', label = 'Inliers')
plt.plot(x[outlier_mask], y[outlier_mask], '.r', label = 'Outlier')
plt.plot(line_x, line_y, '-k', label = 'Linear regression')
plt.plot(line_x, line_y_ransac, '-b', label = 'RANSAC regressor')
plt.legend(loc = 'lower right')
plt.show()





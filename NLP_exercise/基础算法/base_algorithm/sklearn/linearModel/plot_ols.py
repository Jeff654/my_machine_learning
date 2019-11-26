# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


diabetes = datasets.load_diabetes()

# use only one feature
diabetes_x = diabetes.data[:, np.newaxis, 2]

# split the data into trainset and testset
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


# create linear regression object
linear_reg = linear_model.LinearRegression()

# train the model using trainset
linear_reg.fit(diabetes_x_train, diabetes_y_train)

# the coefficients
print('coefficients: \n', linear_reg.coef_)

# the mean square error
print("Residual sum of squares: %.2f" %np.mean((linear_reg.predict(diabetes_x_test) - diabetes_y_test) ** 2))


# explained the variance score: 1 is prefect prediction
print("variance score: %.2f" %linear_reg.score(diabetes_x_test, diabetes_y_test))

# plot outputs
plt.scatter(diabetes_x_test, diabetes_y_test, color = 'black')
plt.plot(diabetes_x_test, linear_reg.predict(diabetes_x_test), color = 'blue', linewidth = 3)

plt.xticks()
plt.yticks()
plt.show()











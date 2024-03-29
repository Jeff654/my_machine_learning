# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
	# function to approximate by polynomial interpolation
	return x * np.sin(x)

# generate points to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset 
x = np.linspace(0, 10, 100)

rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)


# create matrix versions of these arrays
x = x[:, np.newaxis]
x_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color = 'cornflowerblue', linewidth = lw, label = 'ground truth')
plt.scatter(x, y, color = 'navy', s = 30, marker = 'o', label = 'training points')

for count, degree in enumerate([3, 4, 5]):
	model = make_pipeline(PolynomialFeatures(degree), Ridge())
	model.fit(x, y)
	y_plot = model.predict(x_plot)

	plt.plot(x_plot, y_plot, color = colors[count], linewidth = lw, label = "degree %d" %degree)

plt.legend(loc = 'lower left')
plt.show()




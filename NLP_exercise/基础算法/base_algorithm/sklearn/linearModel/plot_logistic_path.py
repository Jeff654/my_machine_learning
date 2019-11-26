# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

iris = datasets.load_iris()
x = iris.data
y = iris.target

x = x[y != 2]
y = y[y != 2]
x -= np.mean(x, 0)


cs = l1_min_c(x, y, loss = 'log') * np.logspace(0, 3)

print("computing regularization path...")
start = datetime.now()
clf = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
coefs_ = []

for c in cs:
	clf.set_params(C = c)
	clf.fit(x, y)
	coefs_.append(clf.coef_.ravel().copy())
print("this took: ", datetime.now() - start)

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_)
ymin, ymax = plt.ylim()

plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('Logistic Regression Path')
plt.axis('tight')
plt.show()





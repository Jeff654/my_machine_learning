# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

n = 500

# 2 latents vars
l1 = np.random.normal(size = n)
l2 = np.random.normal(size = n)

latents = np.array([l1, l1, l2, l2]).T
x = latents + np.random.normal(size = 4 * n).reshape((n, 4))
y = latents + np.random.normal(size = 4 * n).reshape((n, 4))

x_train = x[: n / 2]
y_train = y[: n / 2]
x_test = x[n / 2 :]
y_test = y[n / 2 :]

print("corr(x)")
print(np.round(np.corrcoef(x.T), 2))

print("corr(y)")
print(np.round(np.corrcoef(y.T), 2))

#################################################################
# Canonical (symmetric) PLS
# transform the data
plsca = PLSCanonical(n_components = 2)
plsca.fit(x_train, y_train)
x_train_r, y_train_r = plsca.transform(x_train, y_train)
x_test_r, y_test_r = plsca.transform(x_test, y_test)

# Scatter plot of scores
# ~~~~~~~~~~~~~~~~~~~~~~
# 1) On diagonal plot x vs y scores on each components
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.plot(x_train_r[:, 0], y_train_r[:, 0], "ob", label="train")
plt.plot(x_test_r[:, 0], y_test_r[:, 0], "or", label="test")
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title('Comp. 1: x vs y (test corr = %.2f)' % np.corrcoef(x_test_r[:, 0], y_test_r[:, 0])[0, 1])
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

plt.subplot(224)
plt.plot(x_train_r[:, 1], y_train_r[:, 1], "ob", label="train")
plt.plot(x_test_r[:, 1], y_test_r[:, 1], "or", label="test")
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title('Comp. 2: x vs y (test corr = %.2f)' % np.corrcoef(x_test_r[:, 1], y_test_r[:, 1])[0, 1])
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")

# 2) Off diagonal plot components 1 vs 2 for x and y
plt.subplot(222)
plt.plot(x_train_r[:, 0], x_train_r[:, 1], "*b", label="train")
plt.plot(x_test_r[:, 0], x_test_r[:, 1], "*r", label="test")
plt.xlabel("x comp. 1")
plt.ylabel("x comp. 2")
plt.title('x comp. 1 vs x comp. 2 (test corr = %.2f)' % np.corrcoef(x_test_r[:, 0], x_test_r[:, 1])[0, 1])
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())

plt.subplot(223)
plt.plot(y_train_r[:, 0], y_train_r[:, 1], "*b", label="train")
plt.plot(y_test_r[:, 0], y_test_r[:, 1], "*r", label="test")
plt.xlabel("y comp. 1")
plt.ylabel("y comp. 2")
plt.title('y comp. 1 vs y comp. 2 , (test corr = %.2f)' % np.corrcoef(y_test_r[:, 0], y_test_r[:, 1])[0, 1])
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.show()


##################################################################
# PLS regression, with multivariate response, a.k.a. PLS2
n = 1000
q = 3
p = 10
x = np.random.normal(size = n * p).reshape((n, p))
B = np.array([[1, 2] + [0] * (p - 2)] * q).T

# yj = 1 * x1 + 2 * x2 + noise
y = np.dot(x, B) + np.random.normal(size = n * q).reshape((n, q)) + 5

pls2 = PLSRegression(n_components = 3)
pls2.fit(x, y)
print("True B (such that: y = xB + error)")
print(B)

# compare pls2.coef_ with B
print("Estimated B")
print(np.round(pls2.coef_, 1))
pls2.predict(x)


# PLS regression, with univariate response, a.k.a. PLS1
n = 1000
p = 10
x = np.random.normal(size=n * p).reshape((n, p))
y = x[:, 0] + 2 * x[:, 1] + np.random.normal(size=n * 1) + 5
pls1 = PLSRegression(n_components=3)
pls1.fit(x, y)
# note that the number of compements exceeds 1 (the dimension of y)
print("Estimated betas")
print(np.round(pls1.coef_, 1))

######################################################################
# CCA (PLS mode B with symmetric deflation)

cca = CCA(n_components=2)
cca.fit(x_train, y_train)
x_train_r, y_train_r = plsca.transform(x_train, y_train)
x_test_r, y_test_r = plsca.transform(x_test, y_test)



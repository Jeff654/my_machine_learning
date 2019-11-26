# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

digits = load_digits()
x = digits.images.reshape((len(digits.images), -1))
y = digits.target

# create the RFE object and rank each pixel
svc = SVC(kernel = 'linear', C = 1)
rfe = RFE(estimator = svc, n_features_to_select = 1, step = 1)
rfe.fit(x, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

plt.matshow(ranking)
plt.colorbar()
plt.title('ranking of pixels with RFE')
plt.show()


# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

x = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components = 2)
x_r = pca.fit(x).transform(x)

lda = LinearDiscriminantAnalysis(n_components = 2)
x_r2 = lda.fit(x, y).transform(x)

# percentage of variance explained for each components
print("explained variance ratio (first two components): %s" %str(pca.explained_variance_ratio_))

plt.figure()
for c, i, target_name in zip('rgb', [0, 1, 2], target_names):
	plt.scatter(x_r[y == i, 0], x_r[y == i, 1], c = c, label = target_name)

plt.legend()
plt.title('PCA of IRIS dataset')


plt.figure()
for c, i, target_name in zip('rgb', [0, 1, 2], target_names):
	plt.scatter(x_r2[y == i, 0], x_r2[y == i, 1], c = c, label = target_name)

plt.legend()
plt.title("LDA of IRIS dataset")
plt.show()




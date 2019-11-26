# -*- coding: utf-8 -*-

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection


digits = datasets.load_digits(n_class = 6)
x = digits.data
y = digits.target
n_samples, n_features = x.shape
n_neighbors = 30

# scale and visualize the embedding vectors
def plot_embedding(x, title = None):
	x_min, x_max = np.min(x, 0), np.max(x, 0)
	x = (x - x_min) / (x_max - x_min)

	plt.figure()
	ax = plt.subplot(111)
	for i in range(x.shape[0]):
		plt.text(x[i, 0], x[i, 1], str(digits.target[i]), color = plt.cm.Set1(y[i] / 10.0), fontdict = {'weight': 'bold', 'size': 9})
	
	if hasattr(offsetbox, 'AnnotationBBbox'):
		# only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1.]])
		for i in range(digits.data.shape[0]):
			dist = np.sum((x[i] - shown_images) ** 2, 1)
			if np.min(dist) < 4e-3:
				# don't show points that are too close
				continue
			shown_images = np.r_[shown_images, [x[i]]]
			imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap = plt.cm.gray_r), x[i])
			ax.add_artist(imagebox)
	plt.xticks([])
	plt.yticks([])
	if title is not None:
		plt.title(title)


# plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))

for i in range(n_img_per_row):
	ix = 10 * i + 1
	for j in range(n_img_per_row):
		iy = 10 * j + 1
		img[ix: ix + 8, iy: iy + 8] = x[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap = plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title("a selection from the 64-dimensional digits dataset")


# random 2D projection using a random unitary matirx
print("computing random projection")
rp = random_projection.SparseRandomProjection(n_components = 2, random_state = 42)
x_projected = rp.fit_transform(x)
plot_embedding(x_projected, "random projection of the digits")


# projection on to the first 2 principal components
print("computing PCA projection")
t0 = time()
x_pca = decomposition.TruncatedSVD(n_components = 2).fit_transform(x)
plot_embedding(x_pca, "principal components projection of the digits (time %.2fs)" %(time() - t0))


# projection on to the first 2 linear discriminant components
print("computing linear discriminant analysis projection")
x2 = x.copy()
x2.flat[::x.shape[1] + 1] += 0.01	# make x invertible
t0 = time()
x_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(x2, y)
plot_embedding(x_lda, "Linear Discriminant projection of the digits (time %.2fs)" %(time() - t0))


# Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time()
x_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(x)
print("Done.")
plot_embedding(x_iso, "Isomap projection of the digits (time %.2fs)" %(time() - t0))


# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
t0 = time()
x_lle = clf.fit_transform(x)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(x_lle, "Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))


# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
t0 = time()
x_mlle = clf.fit_transform(x)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(x_mlle, "Modified Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))


# HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
t0 = time()
x_hlle = clf.fit_transform(x)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(x_hlle, "Hessian Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))


# LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
t0 = time()
x_ltsa = clf.fit_transform(x)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(x_ltsa, "Local Tangent Space Alignment of the digits (time %.2fs)" %(time() - t0))


# MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
x_mds = clf.fit_transform(x)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(x_mds, "MDS embedding of the digits (time %.2fs)" % (time() - t0))


# Random Trees embedding of the digits dataset
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
t0 = time()
x_transformed = hasher.fit_transform(x)
pca = decomposition.TruncatedSVD(n_components=2)
x_reduced = pca.fit_transform(x_transformed)

plot_embedding(x_reduced, "Random forest embedding of the digits (time %.2fs)" %(time() - t0))


# Spectral embedding of the digits dataset
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
t0 = time()
x_se = embedder.fit_transform(x)
plot_embedding(x_se, "Spectral embedding of the digits (time %.2fs)" %(time() - t0))


# t-SNE embedding of the digits dataset
# print("Computing t-SNE embedding")
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
# t0 = time()
# x_tsne = tsne.fit_transform(x)
# plot_embedding(x_tsne, "t-SNE embedding of the digits (time %.2fs)" %(time() - t0))
plt.show()



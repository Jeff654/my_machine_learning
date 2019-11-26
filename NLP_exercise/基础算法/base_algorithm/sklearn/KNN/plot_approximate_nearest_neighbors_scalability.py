# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors

n_samples_min = int(1e3)
n_samples_max = int(1e5)
n_features = 100
n_centers = 100
n_queries = 100
n_steps = 6
n_iter = 5

n_samples_values = np.logspace(np.log10(n_samples_min), np.log10(n_samples_max), n_steps).astype(np.int)

# generate some structured data
rng = np.random.RandomState(42)
all_data, _ = make_blobs(n_samples = n_samples_max + n_queries, n_features = n_features, centers = n_centers, shuffle = True, random_state = 0)
queries = all_data[: n_queries]
index_data = all_data[n_queries: ]

# metrics to collect for the plots
average_times_exact = []
average_times_approx = []
accuracies = []

std_times_approx = []
std_accuracies = []
average_speedups = []
std_speedups = []


# calculate the average query time
for n_samples in n_samples_values:
	x = index_data[: n_samples]
	lshf = LSHForest(n_estimators = 20, n_candidates = 200, n_neighbors = 10).fit(x)
	nbrs = NearestNeighbors(algorithm = 'brute', metric = 'cosine', n_neighbors = 10).fit(x)

	time_approx = []
	time_exact = []
	accuracy = []

	for i in range(n_iter):
		# pick one query at random to study query time variability in LSHForest
		query = queries[rng.randint(0, n_queries)]

		t0 = time.time()
		exact_neighbors = nbrs.kneighbors(query, return_distance = False)
		time_exact.append(time.time() - t0)

		##############################################
		t0 = time.time()
		approx_neighbors = lshf.kneighbors(query, return_distance = False)
		time_approx.append(time.time() - t0)

		accuracy.append(np.in1d(approx_neighbors, exact_neighbors).mean())
	
	average_time_exact = np.mean(time_exact)
	average_time_approx = np.mean(time_approx)
	speedup = np.array(time_exact) / np.array(time_approx)

	average_speedup = np.mean(speedup)
	mean_accuracy = np.mean(accuracy)
	std_accuracy = np.std(accuracy)

	print("index size: %d, exact: %0.3fs, LSHF: %0.3fs, speedup: %0.1f, accuracy: %0.2f +/-%0.2f" %(n_samples, average_time_exact, average_time_approx, average_speedup, mean_accuracy, std_accuracy))


	accuracies.append(mean_accuracy)
	std_accuracies.append(std_accuracy)
	average_times_exact.append(average_time_exact)
	average_times_approx.append(average_time_approx)
	std_times_approx.append(np.std(time_approx))
	average_speedups.append(average_speedup)
	std_speedups.append(np.std(speedup))


# plot average query time against n_samples
plt.figure()
plt.errorbar(n_samples_values, average_times_approx, yerr = std_times_approx, fmt = 'o-', c = 'r', label = 'LSHForest')
plt.plot(n_samples_values, average_times_exact, c = 'b', label = 'NearestNeighbors(algorithm = "brute", metric = "cosine")')
plt.legend(loc = "upper left", fontsize = 'small')

plt.ylim(0, None)
plt.ylabel("average query time in seconds")
plt.xlabel("n_samples")
plt.grid(which = 'both')
plt.title("impact of index size on response time for first nearest neighbors queries")


# Plot average query speedup versus index size
plt.figure()
plt.errorbar(n_samples_values, average_speedups, yerr=std_speedups, fmt='o-', c='r')
plt.ylim(0, None)
plt.ylabel("Average speedup")
plt.xlabel("n_samples")
plt.grid(which='both')
plt.title("Speedup of the approximate NN queries vs brute force")

# Plot average precision versus index size
plt.figure()
plt.errorbar(n_samples_values, accuracies, std_accuracies, fmt='o-', c='c')
plt.ylim(0, 1.1)
plt.ylabel("precision@10")
plt.xlabel("n_samples")
plt.grid(which='both')
plt.title("precision of 10-nearest-neighbors queries with index size")

plt.show()

















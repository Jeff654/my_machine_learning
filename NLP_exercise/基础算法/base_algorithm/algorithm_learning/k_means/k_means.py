# -*- coding: utf-8 -*-

from numpy import *


def load_dataSet(filename):
	data_matrix = []
	file = open(filename)

	for line in file.readlines():
		current_line = line.strip().split('\t')

		# 将当前行数据转化成为 float() 型，便于计算
		float_line = map(float, current_line)
		data_matrix.append(float_line)
	
	return data_matrix


def Eclud_distance(vector_A, vector_B):
	return sqrt(sum(power(vector_A - vector_B, 2)))



# 随机选取 k 个质心
def rand_centroid(dataSet, k):
	column = shape(dataSet)[1]
	centroids = mat(zeros((k, column)))

	for j in range(column):
		min_j = min(dataSet[:, j])
		range_j = float(max(dataSet[:, j]) - min_j)
		centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
	return centroids




def k_means(dataSet, k, distance_measure = Eclud_distance, create_centroid = rand_centroid):
	row = shape(dataSet)[0]
	# cluster_assignment 第一列存储记录簇的索引值（该记录隶属于哪个类）
	# 第二列存储该记录到该类质心的距离(存储误差)
	cluster_assignment = mat(zeros((row, 2)))
	centroids = create_centroid(dataSet, k)
	# 用于标识质心是否改变
	cluster_changed = True
	while cluster_changed:
		cluster_changed = False
		for i in range(row):
			min_distance = inf
			min_index = -1
			for j in range(k):
				# 记录 i 到质心 j 的距离
				distance_JI = distance_measure(centroids[j, :], dataSet[i, :])
				if distance_JI < min_distance:
					min_distance = distance_JI
					min_index = j
			if cluster_assignment[i, 0] != min_index:
				cluster_changed = True
			cluster_assignment[i, :] = min_index, min_distance**2
		print "centroids: ", centroids	
		# 重新计算质心
		for cent in range(k):
			# 选取某一个簇内的所有点
			cluster_data = dataSet[nonzero(cluster_assignment[:, 0].A == cent)[0]]
			centroids[cent, :] = mean(cluster_data, axis = 0)
	return centroids, cluster_assignment



# 二分 k_means
# 首先将所有的数据点当做成一个簇，然后将该簇一分为二；然后再选择其中一个簇继续进行划分，选择哪个簇取决于对其划分是否可以最大程度地降低 SSE (sum of square error) 的值。上述基于 SSE 的划分过程不断重复，直到得到用户指定的簇数目为止
def bi_kMeans(dataSet, k, distance_measure = Eclud_distance):
	row = shape(dataSet)[0]

	# 第一列存储聚类标签，第二列存储误差（距离）
	cluster_assignment = mat(zeros((row, 2)))

	# 初始簇
	centroid0 = mean(dataSet, axis = 0).tolist()[0]
	centroid_list = [centroid0]

	for j in range(row):
		cluster_assignment[j, 1] = distance_measure(mat(centroid0), dataSet[j, :])**2

	while (len(centroid_list) < k):
		lowest_SSE = inf
		for i in range(len(centroid_list)):
			current_cluster = dataSet[nonzero(cluster_assignment[:, 0].A == i)[0], :]
			centroid_matrix, split_clusterAssignment = k_means(current_cluster, 2, distance_measure)

			# 计算划分后的误差
			SSE_split = sum(split_clusterAssignment[:, 1])

			# 计算划分前的误差
			SSE_notSplit = sum(cluster_assignment[nonzero(cluster_assignment[:, 0].A != i)[0], 1])

			print "SSE_split, and not_split: ", SSE_split, SSE_notSplit

			if (SSE_split + SSE_notSplit) < lowest_SSE:
				best_centroid_to_split = i

				best_new_centroid = centroid_matrix
				best_cluster_assignment = split_clusterAssignment.copy()
				lowest_SSE = SSE_split + SSE_notSplit

		best_cluster_assignment[nonzero(best_cluster_assignment[:, 0].A == 1)[0], 0] = len(centroid_list)

		best_cluster_assignment[nonzero(best_cluster_assignment[:, 0].A == 0)[0], 0] = best_centroid_to_split

		print "the best_centroid_to_split is: ", best_centroid_to_split
		print "the len of best_cluster_assignment is: ", len(best_cluster_assignment)

		centroid_list[best_centroid_to_split] = best_new_centroid[0, :]
		centroid_list.append(best_new_centroid[1, :])

		cluster_assignment[nonzero(cluster_assignment[:, 0].A == best_centroid_to_split)[0], :] = best_cluster_assignment
	
	return centroid_list, cluster_assignment



# 计算球面距离
def distance_SLC(vector_A, vector_B):
	a = sin(vector_A[0, 1] * pi / 180) * sin(vector_B[0, 1] * pi / 180)
	b = cos(vector_A[0, 1] * pi / 180) * cos(vector_B[0, 1] * pi / 180) * cos(pi * (vector_B[0, 0] - vector_A[0, 0]) / 180)

	return arccos(a + b) * 6371.0


'''

import matplotlib
import matplotlib.pyplot as plt

def cluster_clubs(number_cluster = 5):
	data_list = []
	for line in open('places.txt').readlines():
		line_array = line.split('\t')
		data_list.append([float(line_array[4]), float(line_array[3])])

	data_matrix = mat(data_list)
	
	my_centroids, cluster_assignment = bi_kMeans(data_matrix, number_cluster, distance_measure = distance_SLC)

	fig = plt.figure()
	rect = [0.1, 0.1, 0.8, 0.8]
	scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']

	ax_props = dict(xticks = [], yticks = [])
	ax0 = fig.add_axes(rect, label = 'ax0', **ax_props)

	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)

	ax1 = fig.add_axes(rect, label = 'ax1', frameon = False)

	for i in range(number_cluster):
		current_cluster = data_matrix[nonzero(cluster_assignment[:, 0].A == i)[0], :]
		marker_style = scatter_markers[i % len(scatter_markers)]

		ax1.scatter(current_cluster[:, 0].flatten().A[0], current_cluster[:, 1].flatten().A[0], marker = marker_style, s = 90)
	
	ax1.scatter(my_centroids[:, 0].flatten().A[0], my_centroids[:, 1].flatten().A[0], marker = '+', s = 300)

	plt.show()

'''


import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
	datList = []
	for line in open('places.txt').readlines():
		lineArr = line.split('\t')
		datList.append([float(lineArr[4]), float(lineArr[3])])
	datMat = mat(datList)
	myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distance_SLC)
	fig = plt.figure()
	rect=[0.1,0.1,0.8,0.8]
	scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
	axprops = dict(xticks=[], yticks=[])
	ax0=fig.add_axes(rect, label='ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1=fig.add_axes(rect, label='ax1', frameon=False)
	for i in range(numClust):
		ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
	ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
	plt.show()
















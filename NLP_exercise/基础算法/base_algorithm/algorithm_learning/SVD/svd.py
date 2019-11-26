# -*- coding: utf-8 -*-

from numpy import *
from numpy import linalg as la

def load_dataSet():
	return [[1, 1, 1, 0, 0], 
		[2, 2, 2, 0, 0], 
		[1, 1, 1, 0, 0], 
		[5, 5, 5, 0, 0], 
		[1, 1, 0, 2, 2], 
		[0, 0, 0, 3, 3], 
		[0, 0, 0, 1, 1]]
	
def load_dataSet2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
		[0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
		[0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
		[3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
		[5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
		[0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
		[4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
		[0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
		[0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
		[0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
		[1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 三个度量方法均是列向量
def euclid_similar(vector_A, vector_B):
	return 1.0 / (1.0 + la.norm(vector_A - vector_B))

def pearsion_similar(vector_A, vector_B):
	if len(vector_A) < 3:
		return 1.0
	return 0.5 + 0.5 * corrcoef(vector_A, vector_B, rowvar = 0)[0][1]

def cos_similar(vector_A, vector_B):
	number = float(vector_A.T * vector_B)
	denom = la.norm(vector_A) * la.norm(vector_B)
	return 0.5 + 0.5 * (number / denom)

# 计算用户对商品的估计评分值
def stand_estimate(data_matrix, user, similar_measure, item):
	feature = shape(data_matrix)[1]
	sim_total = 0.0
	rat_simTotal = 0.0
	for j in range(feature):
		user_rating = data_matrix[user, j]
		if user_rating == 0:
			continue

		# 寻找两个用户都评级的商品
		overlap = nonzero(logical_and(data_matrix[:, item].A > 0, data_matrix[:, j].A > 0))[0]
		if len(overlap) == 0:
			similar = 0
		else:
			similar = similar_measure(data_matrix[overlap, item], data_matrix[overlap, j])

		print "the %d and %d similar is: %f" % (item, j, similar)

		sim_total += similar
		rat_simTotal += similar * user_rating
	if sim_total == 0:
		return 0
	else:
		return rat_simTotal / sim_total

def recommend(data_matrix, user, N = 3, similar_measure = cos_similar, estimate_method = stand_estimate):
	unrated_items = nonzero(data_matrix[user, :].A == 0)[1]
	if len(unrated_items) == 0:
		return "you rated everything"
	item_score = []
	for item in unrated_items:
		estimate_score = estimate_method(data_matrix, user, similar_measure, item)
		item_score.append((item, estimate_score))
	return sorted(item_score, key = lambda jj : jj[1], reverse = True)[:N]

def svd_estimate(data_matrix, user, similar_measure, item):
	feature = shape(data_matrix)[1]
	sim_total = 0.0
	rat_simTotal = 0.0

	U, sigma, VT = la.svd(data_matrix)
	sig4 = mat(eye(4) * sigma[:4])

	# 计算变换之后的商品空间（即：V）
	xformed_items = data_matrix.T * U[:, :4] * sig4.I
	for j in range(feature):
		user_rating = data_matrix[user, j]
		if user_rating == 0 or item == j:
			continue
		similar = similar_measure(xformed_items[item, :].T, xformed_items[j, :].T)
		print "the %d and %d similar is: %f" %(item, j, similar)

		sim_total += similar
		rat_simTotal += similar * user_rating
	
	if sim_total == 0:
		return 0
	else:
		return rat_simTotal / sim_total

# 基于 SVD 的图像（32 * 32）压缩
def print_matrix(data_matrix, thresh = 0.8):
	for i in range(32):
		for k in range(32):
			if float(data_matrix[i, k]) > thresh:
				print 1, 
			else:
				print 0, 
		print ""

def image_compress(num_SV = 3, thresh = 0.8):
	myl = []
	for line in open('0_5.txt').readlines():
		new_row = []
		for i in range(32):
			new_row.append(int(line[i]))
		myl.append(new_row)	
	my_data = mat(myl)
	print "***** original matrix *****"
	print_matrix(my_data, thresh)
	U, sigma, VT = la.svd(my_data)
	sig_reconstructer = mat(zeros((num_SV, num_SV)))
	for k in range(num_SV):
		sig_reconstructer[k, k] = sigma[k]	
	reconstructer_matrix = U[:, :num_SV] * sig_reconstructer * VT[:num_SV, :]	
	print "***** reconstructer matrix using %d singular values *****" %num_SV
	print_matrix(reconstructer_matrix, thresh)






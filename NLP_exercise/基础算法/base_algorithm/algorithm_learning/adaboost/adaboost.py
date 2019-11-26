# -*- coding: utf-8 -*-

import numpy

def load_simpData():
	data_matrix = numpy.matrix([[1., 2.1], 
		[2., 1.1], 
		[1.3, 1.], 
		[1., 1.], 
		[2., 1.]])

	class_label = [1.0, 1.0, -1.0, -1.0, 1.0]

	return data_matrix, class_label


# 参数：
	# dimension: 第几维（选取特定的维度）
	# thresh_value: 阈值（维度上的）
	# thresh_inequal: 符号（'lt': less than, 'gt': greater than）
def stump_classify(data_matrix, dimension, thresh_value, thresh_inequal):
	return_array = numpy.ones((data_matrix.shape[0], 1))

	if thresh_inequal == 'lt':
		return_array[data_matrix[:, dimension] <= thresh_value] = -1.0
	else:
		return_array[data_matrix[:, dimension] > thresh_value] = 1.0
	
	return return_array


# weights: 初始权重
def build_stump(data_array, class_label, weights):
	data_matrix = numpy.mat(data_array)
	label_matrix = numpy.mat(class_label).T

	row, column = data_matrix.shape

	num_steps = 10.0
	best_stump = {}
	best_classEst = numpy.mat((row, 1))
	min_error = numpy.inf
	
	# 遍历每一维特征
	for i in range(column):
		range_min = data_matrix[:, i].min()
		range_max = data_matrix[:, i].max()
		step_size = (range_max - range_min) / num_steps

		for j in range(-1, int(num_steps) + 1):
			for inequal in ['lt', 'gt']:
				thresh_value = (range_min + float(j) * step_size)
				predict_value = stump_classify(data_matrix, i, thresh_value, inequal)

				error_array = numpy.mat(numpy.ones((row, 1)))

				error_array[predict_value == label_matrix] = 0

				weights_error = weights.T * error_array

			#	print "split: dim %d, thresh %.2f, thresh inequal: %s, the weights error is %.3f" %(i, thresh_value, inequal, weights_error)

				if weights_error < min_error:
					min_error = weights_error
					best_classEst = predict_value.copy()

					best_stump['dimension'] = i
					best_stump['thresh_value'] = thresh_value
					best_stump['inequal'] = inequal

	return best_stump, min_error, best_classEst



def adaboost_trainDS(data_array, class_label, num_iter = 40):
	weakClass_array = []
	data_array = numpy.mat(data_array)
	row = data_array.shape[0]
	weights = numpy.mat(numpy.ones((row, 1)) / row)

	# 用于累计估计值
	agg_classEst = numpy.mat(numpy.zeros((row, 1)))

	for i in range(num_iter):
		best_stump, min_error, best_classEst = build_stump(data_array, class_label, weights)
	#	print "weights: ", weights.T

		alpha = float(0.5 * numpy.log((1.0 - min_error) / max(min_error, 1e-16)))

		best_stump['alpha'] = alpha
		weakClass_array.append(best_stump)
	#	print "best_classEst: ", best_classEst

		expon = numpy.multiply(-1 * alpha * numpy.mat(class_label).T, best_classEst)
		weights = numpy.multiply(weights, numpy.exp(expon))
		weights = weights / weights.sum()

		agg_classEst += alpha * best_classEst
	#	print "best_classEst: ", best_classEst
		
		agg_error = numpy.multiply(numpy.sign(agg_classEst) != numpy.mat(class_label).T, numpy.ones((row, 1)))

		error_rate = agg_error.sum() / row
		print "total error: ", error_rate, "\n"

		if error_rate == 0.0:
			break
	
	return weakClass_array, agg_classEst


def ada_classify(data_array, classifier_array):
	data_matrix = numpy.mat(data_array)
	row = data_matrix.shape[0]
	agg_classEst = numpy.mat(numpy.zeros((row, 1)))

	for i in range(len(classifier_array)):
		best_classEst = stump_classify(data_matrix, (classifier_array[i])['dimension'], (classifier_array[i])["thresh_value"], (classifier_array[i])['inequal'])
		
		agg_classEst += classifier_array[i]['alpha'] * best_classEst
	#	print "agg_classEst: ", agg_classEst
	
	return numpy.sign(agg_classEst)


def load_dataSet(filename):
	number_feature = len(open(filename).readline().split('\t'))
	data_matrix = []
	label_matrix = []
	file = open(filename)

	for line in file.readlines():
		line_array = []
		current_line = line.strip().split('\t')

		for i in range(number_feature - 1):
			line_array.append(float(current_line[i]))
		
		data_matrix.append(line_array)
		label_matrix.append(float(current_line[-1]))
	
	return data_matrix, label_matrix



# 参数：pred_strengths: 分类器的预测强度
def plot_ROC(pred_strengths, class_label):
	import matplotlib.pyplot as plt
	
	# 记录鼠标光标位置
	cursor = (1.0, 1.0)

	# 用于计算AUC的值
	y_sum = 0.0
	
	# 正例数
	num_posClass = numpy.sum(numpy.array(class_label) == 1.0)

	y_step = 1 / float(num_posClass)
	x_step = 1 / float(len(class_label) - num_posClass)

	sort_index = pred_strengths.argsort()
	
	print "sort_index: ", sort_index

	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)

	for index in sort_index.tolist()[0]:
		if class_label[index] == 1.0:
			delta_x = 0
			delta_y = y_step

		else:
			delta_x = x_step
			delta_y = 0

			y_sum += cursor[1]

		ax.plot([cursor[0], cursor[0] - delta_x], [cursor[1], cursor[1] - delta_y], c = 'b')
		cursor = (cursor[0] - delta_x, cursor[1] - delta_y)

	ax.plot([0, 1], [0, 1], 'b--')

	plt.xlabel("False positive rate")
	plt.ylabel('True positive rate')
	plt.title('ROC curve for adaboost horse colic detection system')

	ax.axis([0, 1, 0, 1])
	plt.show()
	print "the area under the curve is: ", y_sum * x_step




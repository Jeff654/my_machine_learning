# -*- coding: utf-8 -*-

import numpy

def load_dataSet():
	data_matrix = []
	label_matrix = []
	file = open("testSet.txt")

	for line in file.readlines():
		line_array = line.strip().split()
		data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
		label_matrix.append(int(line_array[2]))
	
	return data_matrix, label_matrix


def sigmoid(input_vector):
	return 1.0 / (1 + numpy.exp(-input_vector))


# 批量梯度上升
# 参数：
	# data_inputMatrix：m * n 样本集，m行n列
	# class_label：1 * m 标签
def grad_ascent(data_inputMatrix, class_label):
	data_matrix = numpy.mat(data_inputMatrix)
	label_matrix = numpy.mat(class_label).transpose()
	
#	print "data_matrix: ", data_matrix
#	print "-------------------------------------"
#	print "label_matrix: ", label_matrix

	row, column = data_matrix.shape
	alpha = 0.001

	max_cycles = 5000
	weights = numpy.ones((column, 1))

	for k in range(max_cycles):
		output = sigmoid(data_matrix * weights)
		error = label_matrix - output

		weights = weights + alpha * data_matrix.transpose() * error
	return weights



def plot_bestFit(weights):
	import matplotlib.pyplot as plt
	data_matrix, label_matrix = load_dataSet()
	data_array = numpy.array(data_matrix)

	row = data_array.shape[0]

	x_cord1 = []
	y_cord1 = []
	
	x_cord2 = []
	y_cord2 = []

	for i in range(row):
		if int(label_matrix[i]) == 1:
			x_cord1.append(data_array[i, 1])
			y_cord1.append(data_array[i, 2])
		else:
			x_cord2.append(data_array[i, 1])
			y_cord2.append(data_array[i, 2])

	fig = plt.figure()
	
	ax = fig.add_subplot(111)
	ax.scatter(x_cord1, y_cord1, s = 30, c = 'red', marker = 's')
	ax.scatter(x_cord2, y_cord2, s = 30, c = 'green')

	x = numpy.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	
	y = numpy.array(y.transpose())
	print 'x: ', x
	print x.shape

	print 'y: ', y
	print y.shape

	ax.plot(x, y)

	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()



def stochastic_gradAscent(data_matrix, class_label):
	data_matrix = numpy.mat(data_matrix)
	class_label = numpy.mat(class_label).transpose()
	
	row, column = data_matrix.shape

	alpha = 0.01
	weights = numpy.ones((column, 1))
	
	max_cycle = 200

	for k in range(max_cycle):
		for i in range(row):
			output = sigmoid(sum(data_matrix[i] * weights))
			error = class_label[i] - output

			weights = weights + (alpha * error * data_matrix[i]).transpose()
		print "第 %d 次迭代：" %k
		print "weights = ", weights
	return weights



# 改进的随机梯度上升
def stochastic_gradAscent1(data_matrix, class_label, max_iter = 150):
	data_matrix = numpy.mat(data_matrix)
	class_label = numpy.mat(class_label).transpose()
	
	row, column = data_matrix.shape
	data_index = range(row)

	weights = numpy.ones((column, 1))

	for j in range(max_iter):
		for i in range(row):
			alpha = 4 / (1.0 + i + j) + 0.01
			rand_index = int(numpy.random.uniform(0, len(data_index)))

			output = sigmoid(sum(data_matrix[rand_index] * weights))
			error = class_label[rand_index] - output

			weights = weights + (alpha * error * data_matrix[rand_index]).transpose()
		#	del(data_matrix[rand_index])
		#	del(class_label[rand_index])
		#	del(data_index[rand_index])
	return weights



# 使用logistic回归来预测患有疝病的马的存活率

def classify_vector(input_vector, weights):
	prob = sigmoid(sum(input_vector * weights))

	if prob > 0.5:
		return 1.0
	else:
		return 0.0


def colic_test():
	file_train = open('horseColicTraining.txt')
	file_test = open('horseColicTest.txt')

	training_set = []
	training_label = []

	for line in file_train.readlines():
		current_line = line.strip().split('\t')
		line_array = []
		
		# 每列（特征数）
		for i in range(21):
			line_array.append(float(current_line[i]))

		training_set.append(line_array)
		training_label.append(float(current_line[21]))
	
	train_weights = stochastic_gradAscent1(numpy.array(training_set), training_label, 500)

	error_count = 0
	num_testVector = 0.0

	for line in file_test.readlines():
		num_testVector += 1.0
		current_line = line.strip().split('\t')
		line_array = []

		for i in range(21):
			line_array.append(float(current_line[i]))

		if int(classify_vector(numpy.array(line_array), train_weights)) != int(current_line[21]):
			error_count += 1.0
	error_rate = (float(error_count) / num_testVector)

	print "the error rate of this test is: %f" % error_rate
	return error_rate


def multi_test():
	num_test = 10
	error_sum = 0.0

	for k in range(num_test):
		error_sum += colic_test()
	
	print "after %d iterations, the average error rate is: %f" %(num_test, error_sum / float(num_test))



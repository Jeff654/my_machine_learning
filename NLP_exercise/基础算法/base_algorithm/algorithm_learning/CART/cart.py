# -*- coding: utf-8 -*-

from numpy import *


# param:
	# feature: 待切割的特征
	# value: 待切割特征的特征值
	# right: 右子树
	# left: 左子树
class tree_node():
	def __init__(self, feature, value, right, left):
		feature_split = feature
		value_split = value
		right_branch = right
		left_branch = left



def load_dataSet(filename):
	data_matrix = []
	file = open(filename)

	for line in file.readlines():
		current_line = line.strip().split('\t')

		# 将每行映射成为 float 类型
		float_line = map(float, current_line)
		data_matrix.append(float_line)
	return data_matrix




def binary_splitDataSet(dataSet, feature, value):
	dataSet = mat(dataSet)

	right_matrix = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
	left_matrix = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
	return right_matrix, left_matrix


def reg_leaf(dataset):
	return mean(dataset[:, -1])

def reg_error(dataset):
	return var(dataset[:, -1]) * shape(dataset)[0]



def create_tree(dataSet, leaf_type = reg_leaf, error_type = reg_error, options = (1, 4)):
	feature, value = choose_bestSplit(dataSet, leaf_type, error_type, options)
	if feature == None:
		return value

	ret_tree = {}
	ret_tree['spInd'] = feature
	ret_tree['spVal'] = value

	left_set, right_set = binary_splitDataSet(dataSet, feature, value)
	ret_tree['left'] = create_tree(left_set, leaf_type, error_type, options)
	ret_tree['right'] = create_tree(right_set, leaf_type, error_type, options)

	return ret_tree



def choose_bestSplit(dataSet, leaf_type = reg_leaf, error_type = reg_error, options = (1, 4)):
	total_error = options[0]
	total_number = options[1]

	dataSet = mat(dataSet)

	if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
		return None, leaf_type(dataSet)

	row, column = shape(dataSet)
	error = error_type(dataSet)
	best_error = inf
	best_index = 0
	best_value = 0

	for feature_index in range(column - 1):
		for split_value in set(dataSet[:, feature_index]):
			right_dataSet, left_dataSet = binary_splitDataSet(dataSet, feature_index, split_value)

			if (shape(right_dataSet)[0] < total_number) or (shape(left_dataSet)[0] < total_number):
				continue

			new_error = error_type(right_dataSet) + error_type(left_dataSet)

			if new_error < best_error:
				best_index = feature_index
				best_value = split_value
				best_error = new_error
	if (error - best_error) < total_error:
		return None, leaf_type(dataSet)

	right_dataSet, left_dataSet = binary_splitDataSet(dataSet, best_index, best_value)

	if (shape(right_dataSet)[0] < total_number) or (shape(left_dataSet)[0] < total_number):
		return None, leaf_type(dataSet)

	return best_index, best_value



def is_tree(obj):
	return (type(obj).__name__ == 'dict')

def get_mean(tree):
	if is_tree(tree['right']):
		tree['right'] = get_mean(tree['right'])
	
	if is_tree(tree['left']):
		tree['left'] = get_mean(tree['left'])
	
	return (tree['right'] + tree['left']) / 2.0


def prune(tree, test_data):
	# 无测试数据
	if shape(test_data)[0] == 0:
		return get_mean(tree)
	
	if (is_tree(tree['right']) or is_tree(tree['left'])):
		right_set, left_set = binary_splitDataSet(test_data, tree['spInd'], tree['spVal'])
	
	if is_tree(tree['left']):
		tree['left'] = prune(tree['left'], left_set)
	
	if is_tree(tree['right']):
		tree['right'] = prune(tree['right'], right_set)

	if not is_tree(tree['left']) and not is_tree(tree['right']):
		right_set, left_set = binary_splitDataSet(test_data, tree['spInd'], tree['spVal'])

		# 计算分开的误差
		error_notMerge = sum(power(left_set[:, -1] - tree['left'], 2)) + sum(power(right_set[:, -1] - tree['right'], 2))

		tree_mean = (tree['right'] + tree['left']) / 2.0

		# 计算剪枝（合并）后的误差
		error_merge = sum(power(test_data[:, -1] - tree_mean, 2))

		if error_merge < error_notMerge:
			print "merge"
			return tree_mean

		else:
			return tree
	else:
		return tree



# 模型树：
# 将数据集格式化成目标变量 Y 和自变量 X 
def linear_solve(dataSet):
	row, column = shape(dataSet)
	X = mat(ones((row, column)))
	Y = mat(ones((row, 1)))

	X[:, 1:column] = dataSet[:, 0:column - 1]
	Y = dataSet[:, -1]
	xTx = X.T * X

	if linalg.det(xTx) == 0.0:
		raise NameError('this matrix is singular, cannot do inverse, try increasing the second value of options')

	weights = xTx.I * X.T * Y

	return weights, X, Y



def model_leaf(dataSet):
	weights, X, Y = linear_solve(dataSet)
	return weights

def model_error(dataSet):
	weights, X, Y = linear_solve(dataSet)
	y_hat = X * weights

	return sum(power(Y - y_hat, 2))




def regressTree_evaluate(model, in_data):
	return float(model)


def modelTree_evaluate(model, in_data):
	column = shape(in_data)[1]
	X = mat(ones((1, column + 1)))
	X[:, 1:column + 1] = in_data

	return float(X * model)



def tree_forecast(tree, in_data, model_evaluate = regressTree_evaluate):
	if not is_tree(tree):
		return model_evaluate(tree, in_data)

	if in_data[tree['spInd']] > tree['spVal']:
		if is_tree(tree['left']):
			return tree_forecast(tree['left'], in_data, model_evaluate)
		else:
			return model_evaluate(tree['left'], in_data)
	
	else:
		if is_tree(tree['right']):
			return tree_forecast(tree['right'], in_data, model_evaluate)
		else:
			return model_evaluate(tree['right'], in_data)



def create_forecast(tree, test_data, model_evaluate = regressTree_evaluate):
	row = len(test_data)
	y_hat = mat(zeros((row, 1)))

	for i in range(row):
		y_hat[i, 0] = tree_forecast(tree, mat(test_data[i]), model_evaluate)
	
	return y_hat








# -*- coding: utf-8 -*-

from math import log

def calcShannonEnt(dataSet):
	number_entries = len(dataSet)
	label_count = {}

	for featVec in dataSet:
		current_label = featVec[-1]
		if current_label not in label_count.keys():
			label_count[current_label] = 0
		label_count[current_label] += 1
	
	shannonEnt = 0.0
	for key in label_count:
		prob = float(label_count[key]) / number_entries
		shannonEnt -= prob * log(prob, 2)
	
	return shannonEnt


def createDataSet():
	dataSet = [[1, 1, 'yes'], 
			[1, 1, 'yes'], 
			[1, 0, 'no'], 
			[0, 1, 'no'], 
			[0, 1, 'no']]

	labels = ['no surfacing', 'flippers']

	return dataSet, labels


# 按照属性划分数据集
'''
	@params:
		dataSet: 待划分的数据集
		axis: 待划分的数据集特征
		value: 需要返回的特征的值
'''
def split_dataSet(dataSet, axis, value):
	return_dataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduce_featVec = featVec[:axis]
			reduce_featVec.extend(featVec[axis + 1 :])
			return_dataSet.append(reduce_featVec)
	
	return return_dataSet


# 选择最好的数据集划分方式
def choose_bestFeature_split(dataSet):
	number_features = len(dataSet[0]) - 1
	base_entropy = calcShannonEnt(dataSet)

	best_infoGain = 0.0
	best_feature = -1

	for i in range(number_features):
		feature_list = [example[i] for example in dataSet]
		unique_values = set(feature_list)
		new_entropy = 0.0

		for value in unique_values:
			sub_dataSet = split_dataSet(dataSet, i, value)
			prob = len(sub_dataSet) / float(len(dataSet))
			new_entropy += prob * calcShannonEnt(sub_dataSet)

		info_gain = base_entropy - new_entropy
		if (info_gain > best_infoGain):
			best_infoGain = info_gain
			best_feature = i
	
	return best_feature









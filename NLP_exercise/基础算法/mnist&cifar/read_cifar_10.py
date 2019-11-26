# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np
import os


"""
	this method is aimmed at precessing cifar10 datasets, so you should downlown datasets,
	
	we unzip the datasets, and it includes five data_batch datasets and one test_batch dataset, each of file include 10000 images, 

	and every image is a tensor, which shape is [32, 32, 3]
"""

# load single batch_data
def load_batch_data(file_name):
	with open(file_name, 'rb') as f:
		data_dict = pickle.load(f)
		image_data = data_dict['data']
		image_label = data_dict['labels']

		# image_data = image_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
		image_data = image_data.reshape(10000, 32, 32, 3).astype("float")
		image_label = np.array(image_label)
		
		# 对 size 为 32 * 32 的图像进行裁剪，裁剪后大小为：24 * 24
		image_data = image_data[:, 4 : 28, 4 : 28, :].reshape(10000, 24 * 24 * 3)

		# 对 image_label 进行 one-hot 编码, 编码矩阵为：numSamples * numClasses
		image_label_one_hot = np.zeros((image_label.shape[0], 10))
		for i in range(image_label.shape[0]):
			for j in range(10):
				image_label_one_hot[i][image_label[i]] = 1
		
		return image_data, image_label_one_hot


# load all batch_data and test_batch data
def load_cifar_10(dirctionary):
	images = []
	labels = []
	
	for index in range(1, 6):
		absolute_file = os.path.join(dirctionary, 'data_batch_%d' %(index, ))
		image_data, image_label = load_batch_data(absolute_file)
		
		images.append(image_data)
		labels.append(image_label)
	
	training_image = np.concatenate(images)
	training_label = np.concatenate(labels)

	del image_data, image_label
	
	test_image, test_label = load_batch_data(os.path.join(dirctionary, 'test_batch'))

	return training_image, training_label, test_image, test_label












































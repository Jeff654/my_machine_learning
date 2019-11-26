# -*- codint: utf-8 -*-

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random


image_data = pd.read_csv("image_segment.csv")
image_data = image_data.values			# convert DataFrame to an array


# depart label and features
images = image_data[:, :-1]
labels = image_data[:, -1:]


# convert labels to ont-hot representations
labels_temp = []
for label in labels:
	temp = []
	if label[0] == 'GRASS':
		temp = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	elif label[0] == 'PATH':					
		temp = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	elif label[0] == 'WINDOW':
		temp = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
	elif label[0] == 'CEMENT':
		temp = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
	elif label[0] == 'FOLIAGE':
		temp = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
	elif label[0] == 'SKY':
		temp = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
	else:
		temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
	labels_temp.append(temp)
labels = np.array(labels_temp)


images_temp = []
labels_temp = []

# shuffle the dataset
index_shuffle = [i for i in range(len(labels))]
random.shuffle(index_shuffle)
for i in index_shuffle:
	images_temp.append(images[i])
	labels_temp.append(labels[i])

images = np.array(images_temp)
labels = np.array(labels_temp)


train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size = 0.1)
batch_size = 64
n_banch = len(train_x) // batch_size

X = tf.placeholder(dtype = tf.float32, shape = [None, images.shape[-1]])		# the number of the features
Y = tf.placeholder(dtype = tf.float32, shape = [None, 7])


# define Neural Network
def neural_network():
	w1 = tf.Variable(tf.random_normal([images.shape[-1], 512], stddev = 0.5))
	b1 = tf.Variable(tf.random_normal([512]))
	layer1_out = tf.matmul(X, w1) + b1

	w2 = tf.Variable(tf.random_normal([512, 1024], stddev = 0.5))
	b2 = tf.Variable(tf.random_normal([1024]))
	# layer2_out = tf.nn.relu(tf.matmul(layer1_out, w2) + b2)
	layer2_out = tf.nn.softmax(tf.matmul(layer1_out, w2) + b2)

	w21 = tf.Variable(tf.random_normal([1024, 256], stddev = 0.5))
	b21 = tf.Variable(tf.random_normal([256]))
	layer21_out = tf.nn.softmax(tf.matmul(layer2_out, w21) + b21)

	w3 = tf.Variable(tf.random_normal([256, 7], stddev = 0.5))
	b3 = tf.Variable(tf.random_normal([7]))
	output = tf.nn.softmax(tf.matmul(layer21_out, w3) + b3)

	return output


def train_neural_network():
	output = neural_network()
	cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, Y)))
	learning_rate = tf.Variable(0.001, dtype = tf.float32, trainable = False)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	varibales_list = [variable for variable in tf.trainable_variables()]
	train_step = optimizer.minimize(cost, var_list = varibales_list)

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		
		for epoch in range(100):
			session.run(tf.assign(learning_rate, 0.001 * (0.95 ** epoch)))
			
			for banch in range(n_banch):
				image_banch = train_x[banch * batch_size: (banch + 1) * batch_size]
				label_banch = train_y[banch * batch_size: (banch + 1) * batch_size]
				_, loss = session.run([train_step, cost], feed_dict = {X: image_banch, Y: label_banch})
				print(epoch, banch, loss)


		# precision accuracy
		prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(prediction, dtype = tf.float32))
		accuracy = session.run(accuracy, feed_dict = {X: test_x, Y: test_y})
		print("the precision accuracy is: ", accuracy)


train_neural_network()


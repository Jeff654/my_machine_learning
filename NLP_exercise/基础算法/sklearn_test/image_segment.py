# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

image_data = pd.read_csv("image_segment.csv")
image_data = image_data.values

images = image_data[:, :-1]
labels = image_data[:, -1:]

labels_temp = []
for label in labels:
	temp = []
	if label[0] == "GRASS":
	#	temp = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		temp = 1
	elif label[0] == "PATH":
	#	temp = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		temp = 2
	elif label[0] == "WINDOW":
	#	temp = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
		temp = 3
	elif label[0] == "CEMENT":
	#	temp = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
		temp = 4
	elif label[0] == "FOLIAGE":
	#	temp = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
		temp = 5
	elif label[0] == "SKY":
	#	temp = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
		temp = 6
	else:
	#	temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
		temp = 7
	labels_temp.append(temp)
labels = np.array(labels_temp)

images_temp = []
labels_temp = []

index_shuffle = [i for i in range(len(labels))]
random.shuffle(index_shuffle)
for i in index_shuffle:
	images_temp.append(images[i])
	labels_temp.append(labels[i])
images = np.array(images_temp)
labels = np.array(labels_temp)

train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size = 0.2)

clf = SVC(decision_function_shape = "ovr")
clf.fit(train_x, train_y)

predict_labels = clf.predict(test_x)
# prediction = np.equal(np.argmax(predict_labels, 1), np.argmax(test_y, 1))
# accuracy = 

count = 0
for i in range(len(predict_labels)):
	if np.equal(predict_labels[i], test_y[i]):
		count += 1
accuracy = float(count) / len(test_y)
print("the test precision is: ", accuracy)



















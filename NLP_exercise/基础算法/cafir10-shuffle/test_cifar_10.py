# -*- coding: utf-8 -*-

import tensorflow as tf 
# import tensorflow.examples.tutorials.mnist.input_data as input_data
import read_cifar_10 as rc
# mnist = input_data.read_data_sets('../', one_hot=True) 

x = tf.placeholder(tf.float32, [None, 32 * 32 * 3]) 
y_actual = tf.placeholder(tf.float32, shape=[None, 10]) 

#初始化权值 W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#初始化偏置项 b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#构建卷积层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#构建池化层
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def mean_pool(x):
  return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#构建网络
# x_image = tf.reshape(x, [-1,28,28,1])
training_image, training_label, test_image, test_label = rc.load_cifar_10("cifar-10-batches-py")
x_image = tf.reshape(x, [-1, 32, 32, 3])

W_conv1 = weight_variable([5, 5, 3, 32])      
b_conv1 = bias_variable([32])  

#第一个卷积层   	
h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1) 
#第一个池化层  		
h_pool1 = max_pool(h_conv1)   

print("the shape of pooling 1 is: ", tf.shape(h_pool1), tf.size(h_pool1))                               


W_conv2 = weight_variable([5, 5, 32, 32])
b_conv2 = bias_variable([32])
#第二个卷积层		
h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
#第二个池化层      	
h_pool2 = max_pool(h_conv2)                                   


W_fc1 = weight_variable([8 * 8 * 32, 1024])
b_fc1 = bias_variable([1024])

#reshape成向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 32])  
#第一个全连接层           
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    

keep_prob = tf.placeholder("float") 
#dropout层		
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#softmax层
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  

#交叉熵
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict)) 

#梯度下降法    
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)    
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1)) 

#精确度计算   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 
sess=tf.InteractiveSession()                          
sess.run(tf.initialize_all_variables())

"""
#迭代次数（自己设置）
for i in range(10000):
  batch = mnist.train.next_batch(50)
  #训练100次，验证一次
  if i%100 == 0:                  
    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print('step',i,'training accuracy',train_acc)
  train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("test accuracy",test_acc)
"""


batch_size = 100
batches = training_image.shape[0] // batch_size 

best_accuracy = 0.0

for i in range(500):
	batch = rc.next_batch(50)
	for index in range(len(batch)):
		if index % 100 == 0:
			train_acc = accuracy.eval(feed_dict = {x: batch[index][0], y_actual: batch[index][1], keep_prob: 1.0})

			if train_acc > best_accuracy:
				best_accuracy = train_acc
			print('step', i, 'of index', index, 'accuracy training', train_acc, 'and current best training precision is', best_accuracy)

		train_step.run(feed_dict = {x: batch[index][0], y_actual: batch[index][1], keep_prob: 0.5})

"""
for i in range(100):
	for index in range(batches):
		train_data = training_image[index * batch_size : (index + 1) * batch_size, :]
		train_label = training_label[index * batch_size : (index + 1) * batch_size]
		if index % 100 == 0:
			train_acc = accuracy.eval(feed_dict = {x: train_data, y_actual: train_label, keep_prob: 1.0})
			print('step', i, 'training accuracy: ', train_acc)
		train_step.run(feed_dict = {x: train_data, y_actual: train_label, keep_prob: 0.5})
"""

test_acc = accuracy.eval(feed_dict = {x: test_image, y_actual: test_label, keep_prob: 1.0})
print("test accuracy: ", test_acc)




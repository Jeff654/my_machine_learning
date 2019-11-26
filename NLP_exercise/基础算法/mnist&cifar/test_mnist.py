#coding=utf-8
'''
在上面训练的网络基础上在测试集上进行测试，提取前100个测试图片

'''
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('../', one_hot=True)

import tensorflow as tf

#构建网络部分，与训练过程构建网络部分相同
#可以直接导入train中的函数进行使用

saver = tf.train.Saver()

with tf.Session() as sess:
    #读取上面训练好的模型参数
    saver.restore(sess, 'cnn.model')
    print 'Testing accary: ',sess.run(correct_rate,feed_dict={x:mnist.test.images[:100],y:mnist.test.labels[:100],dropoutP:1.})

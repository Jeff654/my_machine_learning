#coding=utf-8
'''
������ѵ��������������ڲ��Լ��Ͻ��в��ԣ���ȡǰ100������ͼƬ

'''
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('../', one_hot=True)

import tensorflow as tf

#�������粿�֣���ѵ�����̹������粿����ͬ
#����ֱ�ӵ���train�еĺ�������ʹ��

saver = tf.train.Saver()

with tf.Session() as sess:
    #��ȡ����ѵ���õ�ģ�Ͳ���
    saver.restore(sess, 'cnn.model')
    print 'Testing accary: ',sess.run(correct_rate,feed_dict={x:mnist.test.images[:100],y:mnist.test.labels[:100],dropoutP:1.})

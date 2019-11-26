#coding=utf-8
'''
ѵ����һ����������磬����ʶ��mnist
batch_sizeΪ50��ѭ�����й�2000��batch
'''

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('../', one_hot = True)

import tensorflow as tf

image_size = 784 	#28*28
class_num = 10		# 0~10

#��ѵ��100000��
total_step = 100000
#ÿ��10��batch_size����ʾһ�ν��
display_step = 10
#ѧϰ��
learning_rate = 0.01
#ÿ���ҳ�50��ͼƬ����ѵ��
batch_size = 100

#image��placeholder
x = tf.placeholder(tf.float32, [None,image_size])
#label��placeholder
y = tf.placeholder(tf.float32, [None,class_num])
#dropout��placeholder
dropoutP = tf.placeholder(tf.float32)

#����㶨��
def conv2d(image_input,w,b,name):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

#�²����㶨��
def pooling(featuremaps,kernel_size,name):
    return tf.nn.max_pool(featuremaps, [1,kernel_size,kernel_size,1], [1,kernel_size,kernel_size,1], padding='SAME')

#��һ������
def normlize(featuremaps,l_size,name):
    return tf.nn.lrn(featuremaps, 4, bias=1, alpha=0.0001, beta=0.75)

#��ʼ������
weights = {
           #[3,3,1,64]�ֱ����3*3*1��kernel���������1��feature maps������㹲64��feature maps
           'wc1' : tf.Variable(tf.random_normal([3,3,1,64])),
           #[3,3,64,128]�ֱ����3*3*64��kernel���������64��feature maps���������128��feature maps
           'wc2' : tf.Variable(tf.random_normal([3,3,64,128])),
           'wc3' : tf.Variable(tf.random_normal([3,3,128,256])),
           'wc4' : tf.Variable(tf.random_normal([2,2,256,512])),
           #ȫ���Ӳ�Ĳ��������趨��ԭ������Ǿ�������������Ժ�feature map�Ĵ�С��û�з����ı�
           #�����ı��ԭ����pooling��28/2/2/2/2 = 2(7/2������Ϊ4��
           'wd1' : tf.Variable(tf.random_normal([2*2*512,1024])),
           'wd2' : tf.Variable(tf.random_normal([1024,1024])),
           'out' : tf.Variable(tf.random_normal([1024,10]))
           }
#��ʼ��ƫ����
biases = {
          'bc1' : tf.Variable(tf.random_normal([64])),
          'bc2' : tf.Variable(tf.random_normal([128])),
          'bc3' : tf.Variable(tf.random_normal([256])),
          'bc4' : tf.Variable(tf.random_normal([512])),
          'bd1' : tf.Variable(tf.random_normal([1024])),
          'bd2' : tf.Variable(tf.random_normal([1024])),
          'out' : tf.Variable(tf.random_normal([10]))
          }
#��������
def constructNet(images,weights,biases,_dropout):
    #���Ȱ�ͼƬתΪ28*28*1��tensor
    images = tf.reshape(images,[-1,28,28,1])

    #��һ�������conv1
    conv1 = conv2d(images, weights['wc1'], biases['bc1'], 'conv1')
    print 'conv1: ',conv1.get_shape()
    #�����conv1��Ӧ�²�����
    pool1 = pooling(conv1, 2, 'pool1')
    print 'pool1: ',pool1.get_shape()
    #��һ��
    norm1 = normlize(pool1, l_size=4, name='norm1')
    dropout1 = tf.nn.dropout(norm1, _dropout)

    #�ڶ��������
    conv2 = conv2d(dropout1,weights['wc2'],biases['bc2'],'conv2')
    print 'conv2: ',conv2.get_shape()
    pool2 = pooling(conv2, 2, 'pool2')
    print 'pool2: ',pool2.get_shape()
    norm2 = normlize(pool2, 4, 'norm2')
    dropout2 = tf.nn.dropout(norm2,_dropout)

    #�����������
    conv3 = conv2d(dropout2, weights['wc3'], biases['bc3'], 'conv3')
    print 'conv3: ',conv3.get_shape()
    pool3 = pooling(conv3, 2, 'pool3')
    print 'pool3: ',pool3.get_shape()
    norm3 = normlize(pool3, 4, 'norm3')
    dropout3 = tf.nn.dropout(norm3,_dropout)

    #���ĸ������
    conv4 = conv2d(dropout3,weights['wc4'],biases['bc4'],'conv4')
    print 'conv4: ',conv4.get_shape()
    pool4 = pooling(conv4, 2, 'pool4')
    print 'pool4: ',pool4.get_shape()
    norm4 = normlize(pool4, 4, 'norm4')
    print 'norm4: ',norm4.get_shape()
    #ȫ���Ӳ�1
    dense1 = tf.reshape(norm4, [-1,weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1,weights['wd1']) + biases['bd1'],'fc1')
    #ȫ���Ӳ�2
    dense2 = tf.nn.relu(tf.matmul(dense1,weights['wd2']) + biases['bd2'],'fc2')
    #����㣬�������㲻��Ҫ�����relu����
    out = tf.matmul(dense2,weights['out']) + biases['out']
    return out

pred = constructNet(x, weights, biases, dropoutP)
#����loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#���������������С��loss��Adam��һ���ݶ��½��㷨��
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#tf.arg_max(pred,1)�ǰ���ȡ���ֵ���±�
#tf.arg_max(y,1)�ǰ���ȡ���ֵ���±�
correct_pred = tf.equal(tf.arg_max(pred,1), tf.arg_max(y,1))
#�Ƚ�correct_pred�����ݸ�ʽת��Ϊfloat32����
#��correct_pred�е�ƽ��ֵ����Ϊcorrect_pred�г���0����1�������ƽ��ֵ��Ϊ1����ռ����������ȷ��
correct_rate = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step*batch_size < total_step:
        batchx,batchy = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batchx,y:batchy,dropoutP:0.75})
        if(step % display_step == 0):
            accracy = sess.run(correct_rate,feed_dict={x:batchx,y:batchy,dropoutP:1.0})
            cost = sess.run(loss,feed_dict={x:batchx,y:batchy,dropoutP:1.0})
            print 'Step: ' + str(step*batch_size) + ' cost: ' + str(cost) + ' accracy: ' + str(accracy)
            #���浱ǰ����Ĳ������Ա����ʱ��ȡѵ�����
            saver.save(sess, 'cnn.model',step)

        step += 1
    print 'train Finished'
    
    print 'testing accuracy: ', sess.run(correct_rate, feed_dict = {x: mnist.test.images, y: mnist.test.labels, dropoutP: 1.0})

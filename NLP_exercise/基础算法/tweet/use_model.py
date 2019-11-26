# -*- coding: utf-8 -*_


import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# model lexicon.pickle is well-trained
f = open('lexicon.pickle', 'rb')
lex = pickle.load(f)
f.close()


n_input_layer = len(lex)	# input layer
n_layer_1 = 2000		# hidden layer
n_layer_2 = 2000
n_output_layer = 3		# output layer with three classes


def neural_network(data):
	layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_': tf.Variable(tf.random_normal([n_layer_1]))}
	layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_': tf.Variable(tf.random_normal([n_layer_2]))}
	layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_': tf.Variable(tf.random_normal([n_output_layer]))}

	layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
	layer_2 = tf.nn.relu(layer_2)

	layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

	return layer_output


X = tf.placeholder('float')


def prediction(tweet_text):
	predict = neural_network(X)

	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		saver = tf.train.Saver()
		saver.restore(session, 'model.ckpt')

		lemmatizer = WordNetLemmatizer()
		words = word_tokenLimmatize(tweet_text.lower())
		words = [lemmatizer.lemmatize(word) for word in wordds]

		features = np.zeros(len(lex))

		for word in words:
			if word in lex:
				features[lex.index(word)] = 1

		result = session.run(tf.argmax(predict.eval(feed_dict = {X: [features]}), 1))

		return result


prediction("I am very happy") 





































# -*- coding: utf-8 -*-

"""

	使用RNN生成古诗，输入多首古诗词，模型学着生成和前面相关联的字词。

"""

import tensorflow as tf
import numpy as np
import collections

import sys
reload(sys)
sys.setdefaultencoding("utf8")


# ------------------------------- data preprocessing ----------------------------------- #

poetry_file = 'poetry.txt'

# 诗集
poetries = []

with open(poetry_file, 'r') as f:
	for line in f:
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ', '')
			
			if '_' in content or '(' in content or ' (' in content or '《' in content or '[' in content:
				continue
			
			if len(content) < 5 or len(content) > 79:
				continue

			content = '[' + content + ']'
			poetries.append(content)
		
		except Exception as e:
			pass


# 按诗的字数进行排序
poetries = sorted(poetries, key = lambda line: len(line))
print("唐诗总数为: ", len(poetries))


# 统计每个字出现的次数
all_words = []
for poetry in poetries:
	all_words += [word for word in poetry]

counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key = lambda x: -x[1])
words, _ = zip(*count_pairs)


# 选取前n个常用字
words = words[: len(words)] + (' ',)

# 将每个字映射为一个数字 ID
word_num_map = dict(zip(words, range(len(words))))

# 将诗词转换成为向量的形式
word_to_num = lambda word: word_num_map.get(word, len(words))
poetries_to_vector = [list(map(word_to_num, poetry)) for poetry in poetries]		# 每一个向量里的元素为对应诗词的 ID number



# 每次选取 64 首唐诗进行训练
batch_size = 64
n_chunk = len(poetries_to_vector) // batch_size 

x_batches = []
y_batches = []

for i in range(n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size

	batches = poetries_to_vector[start_index: end_index]
	length = max(map(len, batches))

	x_data = np.full((batch_size, length), word_num_map[' '], np.int32)
	for row in range(batch_size):
		x_data[row, :len(batches[row])] = batches[row]
	y_data = np.copy(x_data)
	y_data[:, :-1] = x_data[:, 1:]
	
	"""
	
	x_data				y_data
	[6, 2, 4, 6, 9]			[2, 4, 6, 9, 9]
	[1, 4, 2, 8, 5]			[4, 2, 8, 5, 5]

	"""

	x_batches.append(x_data)
	y_batches.append(y_data)



# ---------------------------------- RNN ------------------------------------------ #
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

def neural_network(model = 'lstm', rnn_size = 128, num_layers = 2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell


	cell = cell_fun(rnn_size, state_is_tuple = True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple = True)

	initial_state = cell.zero_state(batch_size, tf.float32)
	
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words) + 1])
		softmax_b = tf.get_variable("softmax_b", [len(words) + 1])

		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words) + 1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)


	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state = initial_state, scope = 'rnnlm')
	output = tf.reshape(outputs, [-1, rnn_size])

	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	
	return logits, last_state, probs, cell, initial_state



# training the network
def train_neural_network():
	logits, last_state, _, _, _ = neural_network()
	targets = tf.reshape(output_targets, [-1])

	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype = tf.float32)], len(words))
	cost = tf.reduce_mean(loss)

	learning_rate = tf.Variable(0.0, trainable = False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))

	
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		saver = tf.train.Saver(tf.all_variables())

		for epoch in range(50):
			session.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			
			n = 0
			for batche in range(n_chunk):
				train_loss, _, _ = session.run([cost, last_state, train_op], feed_dict = {input_data: x_batches[n], output_targets: y_batches[n]})	
				n += 1
				print(epoch, batche, train_loss)

			if epoch % 7 == 0:
				saver.save(session, 'poetry.model', global_step = epoch)


train_neural_network()



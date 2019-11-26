# -*- codint: utf-8 -*-

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



##############################################################################################

batch_size = 1
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



# --------------------------------- generate poem ----------------------------------#

def generate_poem():
	def to_words(weights):
		t = np.cumsum(weights)
		s = np.sum(weights)

		sampel = int(np.searchsorted(t, np.random.rand(1) * s))
		return words[sample]

	_, last_state, probs, cell, initial_state = neural_network()


	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		
		saver = tf.train.Saver(tf.all_varibales())
		saver.restore(session, 'poetry.model-49')

		state_ = session.run(cell.zero_state(1, tf.float32))
	
		x = np.array([list(map(word_num_map.get, '['))])
		[probs_, state_] = session.run([probs, last_state], feed_dict = {input_data: x, initial_state: state_})
		
		word = to_word(probs_)
		poem = ''
		
		while word != ']':
			poem += word
			x = np.zeros((1, 1))
			x[0, 0] = word_num_map[word]

			[probs_, state_] = session.run([probs, last_state], feed_dict = {input_data: x, initial_state: state_})
			word = to_word(probs_)
			
		return poem

print(generate_poem())










































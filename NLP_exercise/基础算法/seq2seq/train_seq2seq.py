# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.models.rnn.translate import seq2seq_model
# from tensorflow.python.ops import seq2seq
import numpy as np
import math
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""

	使用 seq2seq_model 训练 conversation_to_vector.py 中生成的 train_encoder.vector 文件和 train_decoder.vector 文件

"""

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encoder_vector = 'train_encoder.vector'
train_decoder_vector = 'train_decoder.vector'
test_encoder_vector = 'test_encoder.vector'
test_decoder_vector = 'test_decoder.vector'

# 词汇表大小
vocabulary_encoder_size = 5000
vocabulary_decoder_size = 5000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256		# 每层的神经元个数
num_layers = 3			#  层数
batch_size = 64


# 读取 train/test_encoder.vector 和 train/test_decoder.vector 文件
def read_file(source_path, target_path, max_size = None):
	data_set = [[] for _ in buckets]
	with tf.gfile.GFile(target_path, mode = 'r') as source_file:
		with tf.gfile.GFile(target_path, mode = 'r') as target_file:
			source = source_file.readline()
			target = target_file.readline()
			counter = 0

			while source and target and (not max_size or counter < max_size):
				counter += 1
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(EOS_ID)

				for bucket_id, (source_size, target_size) in enumerate(buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source = source_file.readline()
				target = target_file.readline()
	return data_set


model = seq2seq_model.Seq2SeqModel(source_vocab_size = vocabulary_encoder_size, target_vocab_size = vocabulary_decoder_size, buckets = buckets, size = layer_size, num_layers = num_layers, 
					max_gradient_norm = 5.0, batch_size = batch_size, learning_rate = 0.5, learning_rate_decay_factor = 0.97, forward_only = False)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'		# avoid out of memory

with tf.Session(config = config) as session:
	# 恢复前一次的训练
	ckpt = tf.train.get_checkpoint_state('.')
	
	if ckpt != None:
		print(ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)

	else:
		session.run(tf.global_variables_initializer())

	train_set = read_file(train_encoder_vector, train_decoder_vector)
	test_set = read_file(test_encoder_vector, test_decoder_vector)
	
	train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
	train_total_size = float(sum(train_bucket_sizes))
	train_buckets_scale = [sum(train_bucket_sizes[: i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]


	loss = 0.0
	total_step = 0
	previous_losses = []

	# 一直训练，每段时间保存一次模型
	while True:
		random_number_01 = np.random.sample()
		bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
		
		encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
		_, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
		loss += step_loss / 500
		total_step += 1
		# print("the total size is: ", total_step)

		if total_step % 500 == 0:
			print(model.global_step.eval(), model.learning_rate.eval(), loss)
			
			# 如果模型没有提升，learning_rate 减小
			if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				session.run(model.learning_rate_decay_op)
			previous_losses.append(loss)
			
			# 保存模型
			checkpoint_path = 'chatbot_seq2seq.ckpt'
			model.saver.save(session, checkpoint_path, global_step = model.global_step)
			loss = 0.0

			# 测试评估
			for bucket_id in range(len(buckets)):
				if len(test_set[bucket_id]) == 0:
					continue
				encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
				_, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
				eval_ppx = max.exp(eval_loss) if eval_loss < 300 else float('inf')
				print(bucket_id, eval_ppx)


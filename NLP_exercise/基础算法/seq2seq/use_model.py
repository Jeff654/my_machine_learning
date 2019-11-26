# -*- coding: utf-8 -*-

"""

	使用前面保存的 model

"""
import os
import numpy as np
from tensorflow.models.rnn.translate import seq2seq_model
import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf8')

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encoder_vocabulary = 'train_encoder_vocabulary'
train_decoder_vocabulary = 'train_decoder_vocabulary'

def read_file(input_file):
	temp_vocabulary = []
	with open(input_file, 'r') as f:
		temp_vocabulary.extend(f.readlines())

	temp_vocabulary = [line.strip() for line in temp_vocabulary]
	vocabulary = dict([(x, y) for (y, x) in enumerate(temp_vocabulary)])
	return vocabulary, temp_vocabulary


vocabulary_encoder, _ = read_file(train_encoder_vocabulary)
vocabulary_decoder, _ = read_file(train_decoder_vocabulary)
vocabulary_encoder_size = 5000
vocabulary_decoder_size = 5000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256		# 每层神经元的数量
num_layers = 3			# 网络层数
batch_size = 1

model = seq2seq_model.Seq2SeqModel(source_vocab_size = vocabulary_encoder_size, target_vocab_size = vocabulary_decoder_size, buckets = buckets, size = layer_size, num_layers = num_layers, 
					max_gradient_norm = 5.0, batch_size = batch_size, learning_rate = 0.5, learning_rate_decay = 0.99, forward_only = True)
model.batch_size = 1

with tf.Session() as session:
	# 恢复前一次的训练
	ckpt = tf.train.get_checkpoint_state('.')
	if ckpt != None:
		print(ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("there is not find model.")

	while True:
		input_string = input('me > ')		
		# 退出
		if input_string == 'quit':
			exit()
		
		input_string_vector = []
		for words in input_string.strip():
			input_string_vector.append(vocabulary_encoder.get(words, UNK_ID))
		bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vector)])
		encoder_inputs, decoder_inputs, atrget_weights = model.get_batch({bucket_id: [(input_string_vector, [])]}, bucket_id)
		_, _, output_logits = mpdel.step(session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
		outputs = [int(np.argmax(logit, axis = 1)) for logit in output_logits]

		if EOS_ID == outputs:
			outputs = outputs[: outputs.index(EOS_ID)]
		answer = " ".join([tf.compat.as_str(vocabulary_decoder[output]) for output in outputs])

		print("AI > " + answer)



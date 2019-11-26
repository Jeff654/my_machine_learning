# -*- coding: utf-8 -*-

import os
import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""

	使用影视对白数据集，dgk_shooter_min.conv 是按字分词的语料库
	
"""
conversation_path = "dgk_shooter_min.conv"
if not os.path.exists(conversation_path):
	print("the file is not exist!!!")
	exit()

# 首先将 dgk_shooter_min.conv 文件转换成为 UTF-8 编码格式的
conversations = []
with open(conversation_path) as f:
	one_conversation = []		# 一次完整的对话
	for line in f:
		line = line.strip("\n").replace("/", '')
		if line == '':
			continue
		if line[0] == 'E':		# E 表示分割
			if one_conversation:
				conversations.append(one_conversation)
			one_conversation = []
		elif line[0] == 'M':
			one_conversation.append(line.split(' ')[1])



# 将对话分成问句和答句
asks = []
answers = []
for conversation in conversations:
	if len(conversation) == 1:
		continue
	if len(conversation) % 2 != 0:			# 奇数轮对话，将其转换成偶数轮对话
		conversation = conversation[:-1]	# 直接舍弃最后一句
	for i in range(len(conversation)):		# 遍历对话的轮数
		if i % 2 == 0:
			asks.append(conversation[i])	# 由于下标从 0 开始，偶数轮则表示问句
		else:
			answers.append(conversation[i])


# 使用 seq2seq 的方式
def convert_seq2seq_file(questions, answers, test_size = 8000):
	# 分别创建编码和解码模型的问答文件
	train_encoder = open('train.enc', 'w')		# 问句
	train_decoder = open('train.dec', 'w')		# 答句
	test_encoder = open('test.enc', 'w')		# 问句
	test_decoder = open('test.dec', 'w')		# 答句

	# 选择测试数据
	test_index = random.sample([i for i in range(len(questions))], test_size)
	for i in range(len(questions)):
		if i in test_index:
			test_encoder.write(questions[i] + '\n')
			test_decoder.write(answers[i] + '\n')
		else:
			train_encoder.write(questions[i] + '\n')
			train_decoder.write(answers[i] + '\n')
		if i % 1000 == 0:
			print(len(range(len(questions))), 'the percent is: ', i)

	train_encoder.close()
	train_decoder.close()
	test_encoder.close()
	test_decoder.close()

convert_seq2seq_file(asks, answers)


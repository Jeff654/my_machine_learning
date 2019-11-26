# -*- coding: utf-8 -*-

"""

	将 preprocess_data.py 中生成的 train.enc、train.dec、test_enc、test.dec 文件进行处理：
		首先：创建词汇表
		然后：将对话转换为向量表示 doc2vec (sentence level)

"""
train_encoder_file = 'train.enc'
train_decoder_file = 'train.dec'
test_encoder_file = 'test.enc'
test_decoder_file = 'test.dec'

print("begin to create vocabulary: ")

# 特殊标记，填充标记对话
PAD = '__PAD__'
GO = '__GO__'
EOS = '__EOS__'			# 对话结束的标志
UNK = '__UNK__'			# 标记未登录词（不在词汇表中的词）
START_VOCABULARY = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

vocabulary_size = 5000

print("----------------------------------- generate vocabulary ---------------------------------------------")
# 生成词汇表
def generate_vocabulary_file(input_file, output_file):
	vocabulary = {}
	with open(input_file) as f:
		counter = 0
		for line in f:
			counter += 1
			tokens = [word for word in line.strip()]
			for word in tokens:
				if word in vocabulary:
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1
		
		vocabulary_list = START_VOCABULARY + sorted(vocabulary, key = vocabulary.get, reverse = True)

		# 截取前 vocabulary_size 个常用字符
		if len(vocabulary_list) > vocabulary_size:
			vocabulary_list = vocabulary_list[:vocabulary_size]
		
		print(input_file + ' vocabulary size is: ', len(vocabulary_list))
		
		with open(output_file, 'w') as f_file:
			for word in vocabulary_list:
				f_file.write(word + '\n')	

generate_vocabulary_file(train_encoder_file, 'train_encoder_vocabulary')
generate_vocabulary_file(train_decoder_file, 'train_decoder_vocabulary')

train_encoder_vocabulary_file = 'train_encoder_vocabulary'
train_decoder_vocabulary_file = 'train_decoder_vocabulary'

print("------------          -------------------- generate vector ------------          --------------------------")
# 生成向量
def convert_to_vector(input_file, vocabulary_file, output_file):
	temp_vocabulary = []
	with open(vocabulary_file, 'r') as f:
		temp_vocabulary.extend(f.readlines())
	
	temp_vocabulary = [line.strip() for line in temp_vocabulary]
	vocabulary = dict([(x, y) for (y, x) in enumerate(temp_vocabulary)])
	output_f = open(output_file, 'w')

	with open(input_file, 'r') as f:
		for line in f:
			line_vector = []
			for words in line.strip():
				line_vector.append(vocabulary.get(words, UNK_ID))
			output_f.write(" ".join([str(num) for num in line_vector]) + '\n')
	output_f.close()

convert_to_vector(train_encoder_file, train_encoder_vocabulary_file, 'train_encoder.vector')
convert_to_vector(train_decoder_file, train_decoder_vocabulary_file, 'train_decoder.vector')
convert_to_vector(test_encoder_file, train_encoder_vocabulary_file, 'test_encoder.vector')
convert_to_vector(test_decoder_file, train_decoder_vocabulary_file, 'test_decoder.vector')


# -*- coding: utf-8 -*-

from numpy import *
import feedparser

def loadDataSet():
	posting_list = [
			['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], 
			['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], 
			['my', 'dalmation', 'is', 'so', 'cute', 'i', 'love', 'him'], 
			['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
			['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
			['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
			]
	# 1 代表侮辱性文字，0 代表正常言论
	class_vector = [0, 1, 0, 1, 0, 1]

	return posting_list, class_vector


def create_vocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		# 两个集合的并集
		vocabSet = vocabSet | set(document)
	return list(vocabSet)


# 词集模型：每个词在该文档中只出现一次
def set_words2Vec(vocabList, inputSet):
	return_vector = [0] * len(vocabList)

	for word in inputSet:
		if word in vocabList:
			return_vector[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in my vocabulary" %word
	return return_vector


def train_naiveBayes(train_matrix, train_category):
	# train_matrix: 文档矩阵
	# train_category: 每篇文档类别标签所构成的向量
	num_trainDocs = len(train_matrix)	# 行
	num_words = len(train_matrix[0])	# 列
	
	# 计算侮辱性(1)的文档概率
	pAbusive = sum(train_category) / float(num_trainDocs)
	# p0_num = zeros(num_words)
	# p1_num = zeros(num_words)

	# 避免概率为0，初始化数组为1
	p0_num = ones(num_words)
	p1_num = ones(num_words)
	
	# 总词汇量
	# p0_denom = 0.0
	# p1_denom = 0.0

	p0_denom = 2.0
	p1_denom = 2.0

	for i in range(num_trainDocs):
		if train_category[i] == 1:
			p1_num += train_matrix[i]
			p1_denom += sum(train_matrix[i])
		else:
			p0_num += train_matrix[i]
			p0_denom += sum(train_matrix[i])
	
	# p1_vector = p1_num / p1_denom
	# p0_vector = p0_num / p0_denom

	# 避免多个小概率连乘近似于0的情况，使用对数log计算
	p1_vector = log(p1_num / p1_denom)
	p0_vector = log(p0_num / p0_denom)

	return p0_vector, p1_vector, pAbusive



def classify_naiveBayes(vec2Classify, p0_vector, p1_vector, pClass1):
	p1 = sum(vec2Classify * p1_vector) + log(pClass1)
	p0 = sum(vec2Classify * p0_vector) + log(1.0 - pClass1)

	if p1 > p0:
		return 1
	else:
		return 0


def testing_naiveBayes():
	list_post, list_class = loadDataSet()
	my_vocabList = create_vocabList(list_post)
	train_matrix = []

	for postinDoc in list_post:
		# train_matrix.append(set_words2Vec(my_vocabList, postinDoc))
		train_matrix.append(bagOfWords2Vec(my_vocabList, postinDoc))
	
	p0_vector, p1_vector, pAbusive = train_naiveBayes(array(train_matrix), array(list_class))

	testEntry = ['love', 'my', 'dalmation']
	this_doc = array(set_words2Vec(my_vocabList, testEntry))
	print testEntry, 'classified as: ', classify_naiveBayes(this_doc, p0_vector, p1_vector, pAbusive)

	testEntry = ['stupid', 'garbage']
	this_doc = array(set_words2Vec(my_vocabList, testEntry))
	print testEntry, 'classified as: ', classify_naiveBayes(this_doc, p0_vector, p1_vector, pAbusive)


# 词袋模型：每个单词可能出现多次
def bagOfWords2Vec(vocabList, inputSet):
	return_vector = [0] * len(vocabList)

	for word in inputSet:
		if word in vocabList:
			return_vector[vocabList.index(word)] += 1
	return return_vector


def text_parse(bigString):
	import re
	list_tokens = re.split(r'\w', bigString)

	return [token.lower() for token in list_tokens if len(token) > 2]


def spam_test():
	doc_list = []
	class_list = []
	full_text = []

	for i in range(1, 26):
		word_list = text_parse(open('email/spam/%d.txt' %i).read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(1)

		word_list = text_parse(open('email/ham/%d.txt' %i).read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(0)

	vocabList = create_vocabList(doc_list)
	training_set = range(50)
	test_set = []

	for i in range(10):
		rand_index = int(random.uniform(0, len(training_set)))
		test_set.append(training_set[rand_index])
		del(training_set[rand_index])
	
	train_matrix = []
	train_class = []

	for doc_index in training_set:
		# train_matrix.append(set_words2Vec(vocabList, doc_list[doc_index]))
		train_matrix.append(bagOfWords2Vec(vocabList, doc_list[doc_index]))
		train_class.append(class_list[doc_index])
	
	p0_vector, p1_vector, p_spam = train_naiveBayes(array(train_matrix), array(train_class))
	error_count = 0

	for doc_index in test_set:
		# word_vector = set_words2Vec(vocabList, doc_list[doc_index])
		word_vector = bagOfWords2Vec(vocabList, doc_list[doc_index])

		if classify_naiveBayes(array(word_vector), p0_vector, p1_vector, p_spam) != class_list[doc_index]:
			error_count += 1
	
	print "the error rate is: ", float(error_count) / len(test_set)



def calc_mostFrequency(vocabList, full_text):
	import operator
	freq_dict = {}

	for token in vocabList:
		freq_dict[token] = full_text.count(token)
	
	sorted_freq = sorted(freq_dict.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sorted_freq[:30]


def local_words(feed1, feed0):
	
	doc_list = []
	class_list = []
	full_text = []

	min_length = min(len(feed1['entries']), len(feed0['entries']))

	for i in range(min_length):
		# 每次访问一条RSS源
		word_list = textParse(feed1['entries'][i]['summary'])
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(1)

		word_list = textParse(feed0['entries'][i]['summary'])
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(0)
	
	vocabList = create_vocabList(doc_list)

	top30Words = calc_mostFrequency(vocabList, full_text)

	for pairW in top30Words:
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])
	
	training_set = range(2 * min_length)
	test_set = []

	for i in range(20):
		rand_index = int(random.uniform(0, len(training_set)))
		test_set.append(training_set[rand_index])
		del(training_set[rand_index])
	
	train_matrix = []
	train_class = []

	for doc_index in training_set:
		train_matrix.append(bagOfWords2Vec(vocabList, doc_list[doc_index]))
		train_class.append(class_list[doc_index])
	
	p0_vector, p1_vector, p_spam = train_naiveBayes(array(train_matrix), array(train_class))

	error_count = 0

	for doc_index in test_set:
		word_vector = bagOfWords2Vec(vocabList, doc_list[doc_index])

		if classify_naiveBayes(array(word_vector), p0_vector, p1_vector, p_spam) != class_list[doc_index]:
			error += 1
	
	print "the error rate is: ", float(error_count) / len(test_set)

	return vocabList, p0_vector, p1_vector







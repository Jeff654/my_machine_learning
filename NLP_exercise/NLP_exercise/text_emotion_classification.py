# -*- coding: utf-8 -*-

import collections
import gluonbook as gb
from mxnet import gluon, nd, init
from mxnet.contrib import text
from mxnet.gluon import loss as gloss, data as gdata, nn, rnn, utils as gutils
import random
import tarfile, os
from mxnet_exercise.computer_vision.parameter import WORKING_PATH
import itertools



def download_imdb(data_dir):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, "r") as f:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, data_dir)


def read_imdb(folder="train"):
    """
        每条样本是一条评论以及其对应的标签：1 表示"正面评论"，0 表示"负面评论"
    :param folder:
    :return:
    """
    data = []
    for label in ["pos", "neg"]:
        folder_name = os.path.join(WORKING_PATH + "/NLP_data/imdb_data", folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), "rb") as f:
                review = f.read().decode("utf-8").replace("\n", "").lower()
                data.append([review, 1 if label == "pos" else 0])
    random.shuffle(data)
    return data


def get_token_imdb(data):
    """
        英文语料库：根据空格分词即可(连词暂不考虑)
    :param data:
    :return:
    """
    def tokenizer(text):
        return [token.lower() for token in text.split(" ")]
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    """
        根据已分好的词来尽力词典，并过滤掉频次小于 5 的词汇
    :param data:
    :return:
    """
    tokenized_data = get_token_imdb(data)
    counter = collections.Counter([token for token_list in tokenized_data for token in token_list])
    # counter = collections.Counter(itertools.chain(*tokenized_data))
    return text.vocab.Vocabulary(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    """
        由于每条评论的长度不一，不能直接组合成小批量，因此，对每条评论进行分词，然后
        通过词典转换成为次索引；最后通过截断或者补 0 来进行填充成固定长度
    :param data:
    :param vocab:
    :return:
    """
    max_len = 500
    def padding(x):
        return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))

    tokenized_data = get_token_imdb(data)
    features = nd.array([padding(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])

    return features, labels


def generate_iter_batch_data(train_data, test_data, vocab, batch_size=64):
    """
        本函数旨在生成批量的迭代型数据
    :param train_data:
    :param test_data:
    :param vocab:
    :param batch_size:
    :return:
    """
    train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
    test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
    train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_set, batch_size)

    return train_iter, test_iter


class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        """
            1、在此模型中，每个词先通过嵌入层得到特征向量；
            2、使用双向循环神经网络对特征序列进一步编码，从而得到序列信息；
            3、将编码后的序列信息通过全连接层变换成输出

            将双向长短期记忆在最初时间步和最终时间步的隐藏状态连结，作为特征序列的编码信息
            传递给输出层分类
        :param vocab:
        :param embed_size:
        :param num_hiddens:
        :param num_layers:
        :param kwargs:
        """
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)

        # bidirectional 设置为 True 即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers, bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        """

        :param inputs: shape 为 (batch_size, num_words)；因为 LSTM 需要将序列作为第一维，
                        所以将输入转置后再进行特征提取，输出形状为 (num_words, batch_size, word_vec_dim)
        :return:
        """
        embeddings = self.embedding(inputs.T)

        # states.shape = (num_words, batch_size, 2 * num_hiddens)
        states = self.encoder(embeddings)

        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入，其形状为 (batch_size, 4 * num_hiddens)
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)

        return outputs



def build_LSTM_model(vocab, embed_size, num_hiddens, num_layers, ctx):
    """

    :param vocab:
    :param embed_size:
    :param num_hiddens:
    :param num_layers:
    :param ctx:
    :return:
    """
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.initialize(init.Xavier(), ctx=ctx)

    return net


def predict_sentiment(net, vocab, sentence):
    """
        本函数旨在利用已训练好的模型进行预测
    :param net:
    :param vocab:
    :param sentence:
    :return:
    """
    # sentence = nd.array(vocab.to_indices(sentence), ctx=gb.try_all_gpus())
    sentence = nd.array(vocab.to_indices(sentence))

    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)

    return "positive" if label.asscalar() == 1 else "negative"






if __name__ == "__main__":
    data_dir = WORKING_PATH + r"\NLP_data\imdb_data"
    train_data, test_data = read_imdb(), read_imdb(folder="test")
    print(len(train_data), len(test_data))
    vocab = get_vocab_imdb(train_data)
    print(len(vocab))

    train_iter, test_iter = generate_iter_batch_data(train_data, test_data, vocab)
    # for X, y in train_iter:
    #     print("X: ", X.shape, "y: ", y.shape)
    #     break
    # print(len(train_iter))

    pre_trained_word_vector = WORKING_PATH + r"\NLP_data\glove.6B\glove.6B.100d.txt"

    print(os.path.exists(pre_trained_word_vector))

    glove_embedding = text.embedding.create("glove", pretrained_file_name="glove.6B.100d.txt", vocabulary=vocab)

    ctx = gb.try_all_gpus()
    net = build_LSTM_model(vocab, 100, 100, 2, ctx)
    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.embedding.collect_params().setattr("grad_req", "null")

    lr, num_epochs = 0.05, 10
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

    print(predict_sentiment(net, vocab, ["this", "movie", "is", "so", "great"]))

    print(predict_sentiment(net, vocab, ["this", "movie", "is", "so", "bad"]))










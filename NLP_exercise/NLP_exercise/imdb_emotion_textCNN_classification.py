# -*- coding: utf-8 -*-

import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet_exercise.NLP_exercise.text_emotion_classification import read_imdb


def corr1d(X, K):
    """
        一维卷积
    :param X:
    :param K:
    :return:
    """
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


def corr1d_multi_in(X, K):
    """
        多输入通道一维卷积
    :param X:
    :param K:
    :return:
    """
    # 首先沿着 X 和 Y 的第 0 维(通道维)遍历，使用 * 将结果列表变成 add_n 函数的位置参数来进行相加
    return nd.add_n(*[corr1d(x, k) for x, k in zip(X, K)])


def get_iter_data(batch_size=64):
    """
        获取迭代的训练数据和测试数据
    :param batch_size:
    :return:
    """
    # gb.download_imdb()
    # train_data, test_data = gb.read_imdb("train"), gb.read_imdb("test")

    train_data, test_data = read_imdb("train"), read_imdb("test")

    vocab = gb.get_vocab_imdb(train_data)

    train_iter = gdata.DataLoader(gdata.ArrayDataset(*gb.preprocess_imdb(train_data, vocab)), batch_size=batch_size, shuffle=True)
    test_iter = gdata.DataLoader(gdata.ArrayDataset(*gb.preprocess_imdb(test_data, vocab)), batch_size)

    return vocab, train_iter, test_iter




"""
    TextCNN 主要是用了一维卷积层和时序最大池化层；
    假设输入的文本序列由 n 个词组成，每个词使用 d 维的词向量表示，那么输入样本的宽为 n， 高为 1，输入
    通道数为 d；
    TextCNN steps:
        1、定义多个一维卷积核，并使用这些卷积核对输入分别作卷积计算；宽度不同的卷积核可能会捕捉
           到不同个数的相邻词的相关性
        2、对输出的所有通道分别作时序最大池化，再将这些通道的池化输出值连结为向量
        3、通过全连接层将连结后的向量变换为有关各个类别的输出；此步骤可以使用 dropout-layer 应对 over-fitting

"""
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)

        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)

        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = nn.GlobalAvgPool1D()
        self.convs = nn.Sequential()        # 创建多个一维卷积层

        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation="relu"))

    def forward(self, inputs):
        # 将两个形状 (批量大小，词数，词向量维度)的嵌入层的输出按词向量连结
        embeddings = nd.concat(self.embedding(inputs),
                                self.constant_embedding(inputs),
                                dim=2)

        # 根据 Conv1D 要求的输入格式，将词向量维，即：一维卷积层的通道维，变换到前一维
        embeddings = embeddings.transpose((0, 2, 1))

        # 对每一维卷积层，在时序最大池化后会得到一个形状为 (批量大小，通道大小，1) 的
        # NDArray；使用 flatten 函数去掉最后一维，然后再通道维上进行连结
        encoding = nd.concat(*[nd.flatten(self.pool(conv(embeddings))) for conv in self.convs], dim=1)

        outputs = self.decoder(self.dropout(encoding))

        return outputs


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = gb.try_all_gpus()
vocab, train_data, test_data = get_iter_data()
# train_data, test_data = read_imdb(), read_imdb("test")
# vocab = gb.get_vocab_imdb(train_data)

net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)


glove_embedding = text.embedding.create("glove", pretrained_file_name="glove.6B.100d.txt", vocabulary=vocab)

net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr("grad_req", "null")

lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs)













if __name__ == "__main__":
    # X, K = nd.array([0, 1, 2, 3, 4, 5, 6]), nd.array([1, 2])
    # print(corr1d(X, K))

    X = nd.array([[0, 1, 2, 3, 4, 5, 6],
                  [1, 2, 3, 4, 5, 6, 7],
                  [2, 3, 4, 5, 6, 7, 8]])
    K = nd.array([[1, 2],
                  [3, 4],
                  [-1, -3]])
    print(corr1d_multi_in(X, K))









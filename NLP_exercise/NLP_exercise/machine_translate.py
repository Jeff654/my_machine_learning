# -*- coding: utf-8 -*-

import collections
import io
import math
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
from mxnet_exercise.computer_vision.parameter import WORKING_PATH


PAD, BOS, EOS = "<pas>", "<bos>", "<eos>"

def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    """
        对于一个序列，记录所有的词在 all_tokens 中以便之后构造词典，然后将该序列后添加 PAD
        知道长度变为 max_seq_len，并记录在 all_seqs 中
    :param seq_tokens:
    :param all_tokens:
    :param all_seqs:
    :param max_seq_len:
    :return:
    """
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    """
        使用所有的词来构造词典，并将所有序列中的词变换为次索引后，构造 NDArray 实例
    :param all_tokens:
    :param all_seqs:
    :return:
    """
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens),
                                  reserved_tokens=[PAD, BOS, EOS])
    indicies = [vocab.to_indices(seq) for seq in all_seqs]

    return vocab, nd.array(indicies)


def read_data(max_seq_len):
    """
        使用一份小的 法语-英语 数据集
    :param max_seq_len:
    :return:
    """
    # in 和 out 分别是 input 和 output 的缩写
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    local_file = WORKING_PATH + r"\NLP_data\fr-en-small.txt"
    with io.open(local_file) as f:
        lines = f.readlines()

    for line in lines:
        in_seq, out_seq = line.strip().split("\t")
        in_seq_tokens, out_seq_tokens = in_seq.split(" "), out_seq.split(" ")

        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)

    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)

    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)



class Encoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入为 (批量大小，时间步数); 将输出互换样本维和时间步维
        embedding = self.embedding(inputs).swapaxes(0, 1)
        return self.rnn(embedding, state)

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def attention_model(attention_size):
    """

    :param attention_size:
    :return:
    """
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation="tanh", use_bias=False, flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model


def attention_forward(model, enc_states, dec_state):
    """

    :param model:
    :param enc_states:
    :param dec_state:
    :return:
    """
    # 将解码器的隐藏状态广播到编码器的隐藏状态形状后，再进行连结
    dec_states = nd.broadcast_axis(dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)

    # 形状为：(时间步数, 批量大小, 1)
    e = model(enc_and_dec_states)

    # 在时间步维度上做 softmax 操作
    alpha = nd.softmax(e, axis=0)

    # 返回背景变量
    return (alpha * enc_states).sum(axis=0)



class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, attention_size, drop_prob, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Dense(vocab_size, flatten=False)

    def forward(self, cur_input, state, enc_states):
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[0][-1])

        # 将嵌入后的输入和背景向量在特征维度上进行连结
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)

        # 为输入和背景向量的连结增加时间步维度，时间步个数为 1
        output, state = self.rnn(input_and_c.expand_dims(0), state)

        # 移除时间步维度，输出形状为：(批量大小, 输出词典大小)
        output = self.out(output).squeeze(axis=0)

        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state



def batch_loss(encoder, decoder, out_vocab, X, Y, loss):
    """
        首先实现小批量损失函数：
            1、解码器在最初时间步的输入是特殊字符 BOS
            2、解码器在某一时间步的输入为样本输出序列在上一时间步的词(强制教学)
    :param encoder:
    :param decoder:
    :param out_vocab:
    :param X:
    :param Y:
    :param loss:
    :return:
    """
    batch_size = X.shape[0]
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)

    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是 BOS
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)

    # 使用掩码变量 mask 来忽略掉标签为填充项 PAD 的损失
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size, )), 0
    l = nd.array([0])

    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l += (mask * loss(dec_output, y)).sum()

        # 强制教学
        dec_input = y
        num_not_pad_tokens += mask.sum().asscalar()

        # 当遇到 EOS 时，序列后面的词均将为 PAD，其相应位置的掩码设置为 0
        mask = mask * (y != out_vocab.token_to_idx[EOS])

    return l / num_not_pad_tokens


def train(encoder, decoder, dataset, out_vocab, lr, batch_size, num_epochs):
    """

    :param encoder:
    :param decoder:
    :param dataset:
    :param out_vocab:
    :param lr:
    :param batch_size:
    :param num_epochs:
    :return:
    """
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)

    enc_trainer = gluon.Trainer(encoder.collect_params(), "adam", {"learning_rate": lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), "adam", {"learning_rate": lr})

    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        l_sum = 0
        for X, Y in data_iter:
            with autograd.record():
                l = batch_loss(encoder, decoder, out_vocab, X, Y, loss)
            l.backward()

            enc_trainer.step(1)
            dec_trainer.step(1)

            l_sum += l.asscalar()

        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))









if __name__ == "__main__":
    # max_len_seq = 7
    # in_vocab, out_vocab, dataset = read_data(max_len_seq)
    # print(dataset[0])

    # encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # encoder.initialize()
    # output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))
    #
    # print(output.shape, state[0].shape)
    #
    # # 全连接层仅对输入的最后一维做仿射变换，因此，输出形状中只有最后一维变为全连接层的输出个数 2
    # dense = nn.Dense(2, flatten=False)
    # dense.initialize()
    # print(dense(nd.zeros((3, 5, 7))).shape)
    #
    #
    # seq_len, batch_size, num_hiddens = 10, 4, 8
    # model = attention_model(10)
    # model.initialize()
    # enc_states = nd.zeros((seq_len, batch_size, num_hiddens))
    # dec_state = nd.zeros((batch_size, num_hiddens))
    # print(attention_forward(model, enc_states, dec_state).shape)



    max_len_seq = 7
    in_vocab, out_vocab, dataset = read_data(max_len_seq)
    embed_size, num_hiddens, num_layers = 64, 64, 2
    attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
    encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, drop_prob)
    decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
    train(encoder, decoder, dataset, out_vocab, lr, batch_size, num_epochs)







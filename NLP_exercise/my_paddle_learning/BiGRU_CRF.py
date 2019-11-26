# -*- coding: utf-8 -*-

import paddle
from paddle import fluid
from paddle.dataset import conll05

word_dict, _, label_dict = conll05.get_dict()

word_dim = 32
batch_size = 10
epoch_num = 20
hidden_size = 512
learning_rate = 0.1

word = fluid.layers.data(name="word_data", shape=[1], dtype="int64", lod_level=1)
target = fluid.layers.data(name="target", shape=[1], dtype="int64", lod_level=1)

embedding = fluid.layers.embedding(size=[len(word_dict), word_dim],
                                   input=word,
                                   param_attr=fluid.ParamAttr(name="emb", trainable=False))
hidden_0 = fluid.layers.fc(input=embedding, size=hidden_size, act="tanh")

hidden_1 = fluid.layers.dynamic_lstm(input=hidden_0,
                                     size=hidden_size,
                                     gate_activation="sigmoid",
                                     candidate_activation="relu",
                                     cell_activation="sigmoid")
feature_out = fluid.layers.fc(input=hidden_1, size=len(label_dict), act="tanh")

# 调用内置 CRF 函数，并针对状态转换进行解码
crf_cost = fluid.layers.linear_chain_crf(input=feature_out,
                                         label=target,
                                         param_attr=fluid.ParamAttr(name="crfw", learning_rate=learning_rate))
avg_cost = fluid.layers.mean(crf_cost)
fluid.optimizer.SGD(learning_rate=learning_rate).minimize(avg_cost)


# 声明 PaddlePaddle 计算引擎
place = fluid.CPUPlace()
exe = fluid.Executor(place)
main_program = fluid.default_main_program()
exe.run(fluid.default_startup_program())

# 此脚本为 demo ，直接利用测试集训练model
feeder = fluid.DataFeeder(feed_list=[word, target], place=place)
shuffle_loader = paddle.reader.shuffle(paddle.dataset.conll05.test(), buf_size=8192)
train_data = paddle.batch(shuffle_loader, batch_size=batch_size)

import time
start_time = time.time()
batch_id = 0
for epoch in range(epoch_num):
    for data in train_data():
        data = [[d[0], d[-1]] for d in data]
        cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])

        if batch_id % 10 == 0:
            print("{}/{} epoch avg cost: {}".format(str(epoch), str(epoch_num), str(cost[0][0])))

        batch_id += 1

print("the process during {}".format(time.time() - start_time))









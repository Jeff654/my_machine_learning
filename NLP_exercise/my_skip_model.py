import os
import zipfile
import tensorflow as tf
import numpy as np
import collections
import random
import math
import time

start_time = time.time()


file_path = os.getcwd() + "//file_resource"

"""
    step 1: 读取数据；text8.zip 是包含一个二进制文件，直接使用 zipfile.ZipFile() 进行读取，
            然后再使用 tensorflow 自带的 as_str_any() 将其还原成字符串表示
"""


def read_data(file_name):
    with zipfile.ZipFile(file_name) as f:
        data = tf.compat.as_str_any(f.read(f.namelist()[0])).split()

    return data

words = read_data(file_path + "\\text8.zip")
print("data size: ", len(words))
print(words[:10])


# 选择前 50000 的词频单词，其余的不在列表的使用 unknown 表示
vocabulary_size = 50000


# step 2:
def build_dataset(words):
    """

    :param words:
    :return:
        @data: 单词的 index
        @count: (vocabulary_size + 1) * 2 的二维数组，每个element长度为2，首位表示word，次位表示词频
        @dictionary: 正向的 word->index 的字典
        @reverse_dictionary: 逆向的 index->word 的字典
    """
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        # key 表示当前词，value 表示其在字典中的 索引
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)



"""
    step 3: 生成 batch 数据
    使用 skip-gram，生成 batch-generator
    假设现在有一句”in addition to the function below”，就是将其变为(addition, in)，
    (addition, to)，(to, addition)，(to, the)，(the, to)，(the, function)等，
    假设现在我们只能相邻的两个单词生成样本，也就是说每次是3个单词来生成样本对，
    假设是”addition to the”，我们需要生成”(to, addition)、(to the)”样本对，将其转换为单词的index，
    那么可能是(3, 55)、(3, 89)这样；然后向后滑动，此时3个单词变成”to the function”，
    然后重复上面的步骤
"""


data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    """

    :param batch_size:
    :param num_skips:
    :param skip_window:
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= (2 * skip_window)   # 对每个单词生成多少个样本对

    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1      # [skip_window, target word, skip_window]

    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window        # 单词之间的距离
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)

            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels



# step 4: training
batch_size = 128
embedding_size = 128
skip_window = 128
num_skips = 2
valid_size = 16
valid_window = 100      # 进行valid的单词数量
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64        # 负样本的单词数量

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device("/cpu:0"):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                             labels=train_labels, inputs=embed,
                                             num_sampled=num_sampled, num_classes=vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()

        num_steps = 1000000
        with tf.Session(graph=graph) as session:
            init.run()
            average_loss = 0
            for step in range(num_steps + 1):
                batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
                feed_dict = {
                    train_inputs: batch_inputs,
                    train_labels: batch_labels
                }

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("at steps %d,  and get average loss: %.4f" % (step, average_loss))
                    average_loss = 0

                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 5
                        nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                        log_str = "nearest to %s: " % valid_word

                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s%s, " % (log_str, close_word)

                        print("the word pairs: ", log_str)

                    final_embeddings = normalized_embeddings.eval()


print("*************************************************************************")
print("it last {} seconds".format(str(time.time() - start_time)))

word2vec_file_saved = file_path + "\\word2vec_embedding.txt"
# with open(word2vec_file_saved, "w") as f:
#     f.write(str(final_embeddings))
#     f.close()

np.savetxt(word2vec_file_saved, final_embeddings)



# step 5: 可视化


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_with_label(low_dim_embs, labels, filename="tsne.png"):
    assert low_dim_embs.shape[0] >= len(labels), "more labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                     textcoords="offset points", ha="right", va="bottom")
    plt.savefig(filename)


tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
plot_only = 3000
low_dim_embs = tsne.fit_transform(final_embeddings[: plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_label(low_dim_embs, labels)




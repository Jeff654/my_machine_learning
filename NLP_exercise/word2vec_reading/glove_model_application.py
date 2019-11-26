# -*- coding: utf-8 -*-

import numpy as np
import scipy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import codecs

from word2vec_reading.parameter import WORKING_SPACE

print("the current working space is: ", WORKING_SPACE)

EMBEDDING_FILE = WORKING_SPACE + "/resource/glove.6B/glove.6B.300d.txt"

import os
print(os.path.exists(EMBEDDING_FILE))


def fetch_word_vector(file_name=EMBEDDING_FILE, encoding="utf-8"):
    embeddings_index = {}
    count = 0
    with codecs.open(file_name, encoding=encoding) as f:
        for line in f:
            if count == 0:
                count = 1
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index


def relationship_visual(embedding_vectors, word_list=["king", "queen", "man", "women"]):
    tsne = TSNE(n_components=2)
    word_vectors = []
    for word in word_list:
        word_vectors.append(embedding_vectors[word])

    # 注：python 3 中dict.keys()返回的是一个iterable，but not indexable object
    index_1 = list(embedding_vectors.keys())[:100]
    # all_keys = embedding_vectors.keys()
    # print("the type of keys: ", type(all_keys))
    # print("the content of all keys: ", all_keys)
    # index_1 = all_keys[:100]
    for index in range(len(index_1)):
        word_vectors.append(embedding_vectors[index_1[index]])
    word_vectors = np.array(word_vectors)
    words_tsne = tsne.fit_transform(word_vectors)

    ax = plt.subplot(111)
    for index in range(len(word_list)):
        plt.text(words_tsne[index, 0], word_vectors[index, 1], word_list[index])
        plt.xlim((-100, 20))
        plt.ylim((-50, 50))
    plt.show()



if __name__ == "__main__":
    embedding_vectors = fetch_word_vector()
    # print("found {0} word vectors of glove.".format(len(embedding_vectors)))

    # king_wordvec = embedding_vectors["king"]
    # queen_wordvec = embedding_vectors["queen"]
    # man_wordvec = embedding_vectors["man"]
    # women_wordvec = embedding_vectors["women"]
    #
    # pseudo_king = queen_wordvec - women_wordvec + man_wordvec
    # cosine_similar = np.dot(pseudo_king / np.linalg.norm(pseudo_king), king_wordvec / np.linalg.norm(king_wordvec))
    # print("the cosine similarity: ", cosine_similar)


    relationship_visual(embedding_vectors)






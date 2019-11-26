# -*- coding: utf-8 -*-

import numpy as np


corpus = ["i like machine learning.",
          "i like tensorflow.",
          "i prefer python."]

corpus_words_unique = set()
corpus_processed_docs = []
for doc in corpus:
    corpus_words_ = []
    corpus_words = doc.split()
    for word in corpus_words:
        if len(word.split(".")) == 2:
            corpus_words_ += [word.split(".")[0]] + ["."]
        else:
            corpus_words_ += word.split(".")
    corpus_processed_docs.append(corpus_words_)
    corpus_words_unique.update(corpus_words_)

corpus_words_unique = np.array(list(corpus_words_unique))
co_occurence_matrix = np.zeros((len(corpus_words_unique), len(corpus_words_unique)))
for corpus_words_ in corpus_processed_docs:
    for index in range(1, len(corpus_words_)):
        index_1 = np.argwhere(corpus_words_unique == corpus_words_[index])
        index_2 = np.argwhere(corpus_words_unique == corpus_words_[index - 1])

        co_occurence_matrix[index_1, index_2] += 1
        co_occurence_matrix[index_2, index_1] += 1

u, s, v = np.linalg.svd(co_occurence_matrix, full_matrices=False)
print("co_occurence_matrix follows: \n", co_occurence_matrix)

import matplotlib.pyplot as plt
for index in range(len(corpus_words_unique)):
    plt.text(u[index, 0], u[index, 1], corpus_words_unique[index])
plt.xlim((-0.75, 0.75))
plt.ylim((-0.75, 0.75))

plt.show()







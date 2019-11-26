# -*- coding: utf-8 -*-

from gensim import corpora, models
from gensim.models import Word2Vec, FastText
from gensim.test.utils import common_texts, get_tmpfile

path = get_tmpfile("word2vec.model")

# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4, hs=1)
# model.save("word2vec.model")

model = FastText(common_texts, size=4, window=3, min_count=1, iter=10)
model.save("fasttext.model")


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from scipy.special import expit
#     x = np.linspace(-6, 6, 121)
#     y = expit(x)
#     plt.plot(x, y)
#     plt.grid()
#     plt.xlim(-6, 6)
#     plt.xlabel("x")
#     plt.title("expit(x)")
#     plt.show()


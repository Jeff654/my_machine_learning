# -*- coding: utf-8 -*-

from mxnet import nd
from mxnet.contrib import text

# print(text.embedding.get_pretrained_file_names().keys())
# print(text.embedding.get_pretrained_file_names("glove"))

glove_6b_50d = text.embedding.create("glove", pretrained_file_name="glove.6B.50d.txt")
# print(len(glove_6b_50d))
print(glove_6b_50d.token_to_idx["beautiful"], glove_6b_50d.idx_to_token[3367])


def knn(W, x, k):
    """
        使用余弦相似度来搜索近义词
    :param W:
    :param x:
    :param k:
    :return:
    """
    cos = nd.dot(W, x.reshape((-1, ))) / (nd.sum(W * W, axis=1).sqrt() * nd.sum(x * x).sqrt())
    top_k = nd.topk(cos, k=k, ret_typ="indices").asnumpy().astype("int32")
    return top_k, [cos[i].asscalar() for i in top_k]

def get_similar_tokens(query_token, k, embed_words):
    """
        通过预训练的词向量实例 embed_words 来搜索近义词
    :param query_token:
    :param k:
    :param embed_words:
    :return:
    """
    top_k, cos = knn(embed_words.idx_to_vec, embed_words.get_vecs_by_tokens([query_token]), k + 2)
    print([embed_words.idx_to_token[index] for index in top_k])
    for i, c in zip(top_k[2:], cos[2:]):
        # 除去输入词和未知词
        print("cosine sim=%.3f: %s" % (c, (embed_words.idx_to_token[i])))


def get_analogy(token_a, token_b, token_c, embed_words):
    """
        求类彼此：a is to b just like c is to ...
    :param token_a:
    :param token_b:
    :param token_c:
    :param embed_words:
    :return:
    """
    vecs = embed_words.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    top_k, cos = knn(embed_words.idx_to_vec, x, 2)
    print([embed_words.idx_to_token[index] for index in top_k])
    return embed_words.idx_to_token[top_k[0]]       # 剔除未知值








if __name__ == "__main__":
    # get_similar_tokens("beautiful", 6, glove_6b_50d)
    print(get_analogy("man", "women", "son", glove_6b_50d))







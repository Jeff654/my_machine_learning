# -*- coding: utf-8 -*-

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle



import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from mxnet_exercise.computer_vision.parameter import WORKING_PATH



def ingest(file_name):
    """
        define a function that loads data set adn extracts the two columns we need
            1、the sentiment a binary (0/1) variable
            2、the text of tweet string
    :return:
    """
    data = pd.read_csv(file_name, encoding="latin-1")
    data.drop(["ItemID", "SentimentSource"], axis=1, inplace=True)
    data = data[data.Sentiment.isnull() == False]
    data["Sentiment"] = data["Sentiment"].map(int)

    data = data[data["SentimentText"].isnull() == False]
    data.reset_index(inplace=True)
    data.drop("index", axis=1, inplace=True)

    return data


def tokenize(tweet):
    """
        split each tweet into tokens and removes user mentions, hashtags and
        urls, these elements are very common in tweets but unfortunately they do
        not provide enough semantic information for the task
    :param tweet:
    :return:
    """
    try:
        # tweet = tweet.decode("utf-8").lower()
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)

        tokens = filter(lambda t: not t.startswith("@"), tokens)
        tokens = filter(lambda t: not t.startswith("#"), tokens)
        tokens = filter(lambda t: not t.startswith("http"), tokens)
        tokens = list(tokens)

        return tokens
    except:
        return "NC"


def post_process(data, n=1000000):
    """
        取前 n 条数据
    :param data:
    :param n:
    :return:
    """
    data = data.head(n)
    data["tokens"] = data["SentimentText"].progress_map(tokenize)
    data = data[data.tokens != "NC"]
    data.reset_index(inplace=True)
    data.drop("index", inplace=True, axis=1)

    return data


def split_data_set(data, n=1000000):
    """
        split the training set and test set
    :param data:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                        np.array(data.head(n).Sentiment), test_size=0.2)

    def labelizeTweets(tweets, label_type):
        """

        :param tweets:
        :param label_type:
        :return:
        """
        labelized = []
        for index, value in tqdm(enumerate(tweets)):
            label = "%s_%s" % (label_type, index)
            labelized.append(LabeledSentence(value, [label]))
        return labelized

    x_train = labelizeTweets(x_train, "TRAIN")
    x_test = labelizeTweets(x_test, "TEST")

    return x_train, x_test, y_train, y_test


def build_word_vector(wordVec, tfidf_dict, tokens, size):
    """

    :param wordVec:
    :param tfidf_dict:
    :param tokens:
    :param size:
    :return:
    """
    vector = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vector += wordVec[word].reshape((1, size)) * tfidf_dict[word]
            count += 1
        except KeyError:
            continue

    if count != 0:
        vector /= count

    return vector








if __name__ == "__main__":
    file_name = WORKING_PATH + r"\NLP_data\tweet_sentiment\tweet_data\train.csv"
    data = ingest(file_name)
    # print(data.head(5))

    data = post_process(data)
    train_data, test_data, train_label, test_label = split_data_set(data)
    print("*************************************************")
    print(train_data)
    print("--------------------------------------------------------")
    print(train_label)


    n_dim = 200
    tweet_w2v = Word2Vec(size=n_dim, min_count=10)

    tweet_w2v.build_vocab([x.words for x in tqdm(train_data)])
    tweet_w2v.train([x.words for x in tqdm(train_data)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)
    # print(tweet_w2v["good"])
    # print(tweet_w2v.most_similar("good"))
    # print(tweet_w2v.most_similar("bar"))
    # print(tweet_w2v.most_similar("facebook"))
    # print(tweet_w2v.most_similar("iphone"))


    import bokeh.plotting as bp
    import bokeh
    from bokeh.models import HoverTool, BoxSelectTool
    from bokeh.plotting import figure, show, output_notebook

    output_notebook()
    plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                           title="A map of 10000 word vectors",
                           tools="pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                           x_axis_type=None, y_axis_type=None, min_border=1)

    # get a list of word vectors, limit to 10000, each is of 200 dimensions
    word_vectors = [tweet_w2v[word] for word in list(tweet_w2v.wv.vocab.keys())[:5000]]

    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_w2v = tsne_model.fit_transform(word_vectors)

    # put everything in a dataframe
    tsne_tf = pd.DataFrame(tsne_w2v, columns=["x", "y"])
    tsne_tf["words"] = list(tweet_w2v.wv.vocab.keys())[:5000]

    # print(tsne_tf)
    # print(type(tsne_tf))

    tsne_tf = bokeh.models.sources.ColumnDataSource(tsne_tf)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(type(tsne_tf))

    # plot
    plot_tfidf.scatter(x="x", y="y", source=tsne_tf)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips = {"word": "@words"}
    show(plot_tfidf)


    # weighted average: tf-idf
    print("building tf-idf matrix...", )
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in train_data])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print("vocab size: ", len(tfidf))
    print(tfidf)


    from sklearn.preprocessing import scale
    train_vecs_w2v = np.concatenate([build_word_vector(tweet_w2v, tfidf, tokens, n_dim) for tokens in tqdm(map(lambda x: x.words, train_data))])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([build_word_vector(tweet_w2v, tfidf, tokens, n_dim) for tokens in tqdm(map(lambda x: x.words, test_data))])
    test_vecs_w2v = scale(test_vecs_w2v)




    from mxnet.gluon.nn import Sequential, Dense
    # from mxnet import autograd, gluon, init, nd
    #
    #
    # model = Sequential()
    # model.add(Dense(32, activation="relu", input_dim=200))
    # model.add(Dense(1, activation="sigmoid"))
    # model.initialize(init=init.Xavier())
    # trainer = gluon.Trainer(model.collect_params(), "sgd", {"learing_rate": 0.01})


    # model = Sequential()
    # model.add(Dense(32, activation="relu", input_dim=200))
    # model.add(Dense(1, activation="sigmoid"))
    # model.compile(optimizer="rmsprop",
    #               loss="binary_crossentropy",
    #               metrics=["accuracy"])
    # model.fit(train_vecs_w2v, train_label, epochs=9, batch_size=32, verbose=2)
    # score = model.evaluate(test_vecs_w2v, test_label, batch_size=128, verbose=2)
    # print(score[1])






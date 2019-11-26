# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


# 方法一：基本使用方法，即使用前面分类器产生的特征特征输出作为最后总得meta-classifier的输入数据
def train():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

    for clf, label in zip([clf1, clf2, clf3, sclf], ["KNN", "random forest", "naive bayes", "stacking classifier"]):
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



# 方法二：使用第一层基本分类器产生的类别概率值作为meta-classifier的输入
# 注：需要将StackingClassifier的参数设置 use_probas=True, 如若设置average_probas=True，那么
# 这些及分类器对每一个类别产生的概率值会被平均，否则，进行拼接

"""
    加入有两个基分类器产生的概率输出为：
    model1: [0.2, 0.5, 0.3]
    model2: [0.3, 0.4, 0.3]
    
    a) average_probas = True
        meta_feature: [0.25, 0.45, 0.3]
    b) average_probas = False
        meta_feature: [0.2, 0.5, 0.3, 0.3, 0.4, 0.3]
"""


def train2():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              use_probas=True,
                              average_probas=False,
                              meta_classifier=lr)

    for clf, label in zip([clf1, clf2, clf3, sclf], ["KNN", "random forest", "naive bayes", "stacking classification"]):
        scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



"""
    方法三：对训练集中的特征维度进行操作
        此处不是给每一个 base-classifier 全部的特征，而是给定不同的 base-classifier 不同的特征；
        如：model1 训练前半部分的特征，model2训练后半部分的特征(可通过sklearn的pipelines实现)，
        然后再通过 StackingClassifier 组合起来
"""


def train3():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)), LogisticRegression())
    pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)), LogisticRegression())

    sclf = StackingClassifier(classifiers=[pipe1, pipe2], meta_classifier=LogisticRegression())

    sclf.fit(x, y)





if __name__ == "__main__":
    train2()







# -*- coding: utf-8 -*-

"""
    本脚本旨在针对model进行stacking的各种相关操作
    知乎链接：https://www.zhihu.com/search?type=content&q=blending
"""
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np



# 回归问题
N_FOLDS = 5         # 采用5折交叉验证
kf = KFold(n_splits=N_FOLDS, random_state=42)


def get_oof(clf, x_train, y_train, x_test):
    """
        本函数旨在针对回归问题进行单模型clf的单层训练
        eg:
            训练集有1000个samples，10个features，测试集有500个samples
            x_train: 1000 * 10
            y_train: 1 * 1000
            x_test = 500 * 10

        诸如 n_folds = 5
        step1: 从 n_folds - 1 折数据(80%的数据)中利用 clf 训练出一个model，然后预测剩下的一折(20%的数据)，同时预测测试集
        step2: 每次均存在20%的训练数据被预测，5次后正好每个样本都被预测过了
        step3: 每次均要预测测试集，最后进行平均

        details:
            step1: 每次fold，生成800个train data，200个valid data，利用800个train data训练model1，然后
                    预测200个valid data，预测结果为程度200的预测值
            step2: step1 走 n_folds 次，长度为 200 的预测值也有5个，亦即：200 * 5 = 1000，刚好所有的数据
                    均有被预测到；注意：此1000个数据是由model1产生的，先存着，作为第二层model的来源数据
                注：step2 产生的预测值为：P1 = 1000 * 1
            step3: 针对500个test data，每次fold，model均要去预测所有的500个test data，预测结果为长度为
                    500 的预测值；走 n_folds 次，可以得到一个 n_folds * 500 的预测值矩阵，然后根据行取
                    平均值即可，得到 p1 = 1 * 500 的平均预测值

            注：第一层如若有其他的model，则只需变换model即可，其与操作不变；
            PS: 假设第一层存在三个model，则得到一个来自训练数据集的预测值矩阵：1000 * 3，以及一个来自
                测试数据集的预测值矩阵：500 * 3

            第二层：将第一层中来自training data的预测值矩阵作为训练数据，训练第二层model
                    将第一层中来自testing data的预测值矩阵作为测试数据，测试第二层model
            第三层...


    :param clf: model object
    :param x_train: all training data
    :param y_train: all training labels
    :param x_test: all testing data
    :return:
    """
    oof_train = np.zeros((x_train.shape[0], 1))     # stacking后训练数据的输出
    # oof_test_skf[i]表示第 i 折交叉验证产生的model对测试机的预测结果
    oof_test_skf = np.empty((N_FOLDS, x_test.shape[0], 1))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        kf_x_train = x_train[train_index]       # 800 * 10 训练集
        kf_y_train = y_train[train_index]       # 1 * 800 训练集对应的输出
        kf_x_val = x_train[test_index]          # 200 * 10 验证集

        clf.fit(kf_x_train, kf_y_train)         # 利用当前model进行训练

        # 对当前验证集进行预测    200 * 1
        oof_train[test_index] = clf.predict(kf_x_val).reshape(-1, 1)

        # 对测试集进行预测：oof_test_skf[i, :].shape = (500, 1)
        oof_test_skf[i, :] = clf.predict(x_test).reshape(-1, 1)

    # 针对测试集上的表现结果进行求均值
    oof_test = oof_test_skf.mean(axis=0)

    return oof_train, oof_test




# 如若是分类问题，则需要修改get_oof()
N_CLASS = 2


def get_classification_oof(clf, x_train, y_train, x_test):
    """

    :param clf:
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    """
    oof_train = np.zeros((x_train.shape[0], N_CLASS))
    oof_test = np.empty((x_test.shape[0], N_CLASS))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_test = x_train[test_index]

        clf.fit(kf_x_train, kf_y_train)

        oof_train[test_index] = clf.predict_proba(kf_x_test)
        oof_test += clf.predict_proba(x_test)

    oof_test /= N_CLASS

    return oof_train, oof_test












if __name__ == "__main__":
    # 生成数据
    x_train = np.random.random((1000, 10))
    y_train = np.random.random_integers(0, 1, (1000, ))
    x_test = np.random.random((500, 10))

    new_train, new_test = [], []
    for clf in [LinearRegression(), RandomForestRegressor()]:
        oof_train, oof_test = get_oof(clf, x_train, y_train, x_test)
        new_train.append(oof_train)
        new_test.append(oof_test)

    new_train = np.concatenate(new_train, axis=1)
    new_test = np.concatenate(new_test, axis=1)

    # 利用新的训练数据 new_train 作为新的model的输入，stacking 第二层
    clf = RandomForestRegressor()
    clf.fit(new_train, y_train)
    clf.predict(new_test)












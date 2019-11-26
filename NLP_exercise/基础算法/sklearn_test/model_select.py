# -*- coding: utf-8 -*-

"""
	TPOT 是基于遗传算法的自动选择，优化机器学习模型和参数的工具。

"""
from sklearn import model_selection
from tpot import TPOTClassifier
import numpy as np
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', np.nan, inplace = True)
df.dropna(inplace = True)
df.drop(['id'], 1, inplace = True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)

tpot = TPOTClassifier(generations = 6, verbosity = 2)
tpot.fit(X_train, Y_train)
tpot.score(X_test, Y_test)

tpot.export('pipeline1.py')








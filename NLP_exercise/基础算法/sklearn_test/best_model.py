# -*- coding: utf-8 -*-

"""
	model_select.py	中 TPOTClassifier 选择了 ExtraTreesClassifer 作为最准确的分类器

"""

from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?', np.nan, inplace = True)
df.dropna(inplace = True)
df.drop(['id'], 1, inplace = True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)
# clf = ExtraTreesClassifier(criterion = 'gini', max_features = 0.3, n_estimators = 500)
clf = GradientBoostingClassifier(criterion = 'mse', max_features = 0.3, n_estimators = 500)

clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)






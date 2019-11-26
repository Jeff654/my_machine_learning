# -*- coding: utf-8 -*-

import numpy as np
from numpy import column_stack, loadtxt
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import time
start_time = time.time()

data = read_csv("XGBoost_resource/breast-cancer.data", delimiter=",")
data = data.values

X = data[:, 0:9]
X = X.astype(str)
Y = data[:, 9]

columns = []
label_encoder = LabelEncoder()
for index in range(0, X.shape[1]):
    # label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(X[:, index])
    feature = feature.reshape(X.shape[0], 1)

    one_hot_encoder = OneHotEncoder(sparse=False, categories="auto")
    feature = one_hot_encoder.fit_transform(feature)
    columns.append(feature)

encoded_x = column_stack(columns)
print("X shape: ", encoded_x.shape)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)

seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(encoded_x, label_encoded_y,
                                                    test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)
print("accuracy: %.4f%%" % (accuracy * 100.0))

print("the procedure lasts: ", time.time() - start_time)








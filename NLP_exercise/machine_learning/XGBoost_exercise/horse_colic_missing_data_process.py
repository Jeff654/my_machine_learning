# -*- coding: utf-8 -*-

import numpy as np
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Imputer

data_frame = read_csv("XGBoost_resource/horse-colic.data", delim_whitespace=True, header=None)
data_set = data_frame.values

X = data_set[:, 0:27]
Y = data_set[:, 27]

# set missing values to 0
# X[X == "?"] = 0
X[X == "?"] = np.nan

# convert str type to numeric
X = X.astype("float32")

# impute = Imputer()
# impute = Imputer(strategy="most_frequent")
# impute_x = impute.fit_transform(X)


label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)

seed = 7
test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X, label_encoded_y,
                                                    test_size=test_size, random_state=seed)
# X_train, X_test, Y_train, Y_test = train_test_split(impute_x, label_encoded_y,
#                                                     test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)
print("accuracy: %.4f%%" % (accuracy * 100.0))

















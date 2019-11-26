# -*- coding: utf-8 -*-

# multi-class classification
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = read_csv("XGBoost_resource/Iris.csv", header=None)
data_set = data.values

X = data_set[1:, 1:5]
Y = data_set[1:, 5]

# label encoder
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)

seed = 7
test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X, label_encoded_y, test_size=test_size,
                                                    random_state=seed)

model = XGBClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)
print("accuracy: %.4f%%" % (accuracy * 100.0))















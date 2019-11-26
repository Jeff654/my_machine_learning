# -*- coding: utf-8 -*-

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib


data_set = loadtxt("XGBoost_resource/pima-indians-diabetes.csv", delimiter=",")

X = data_set[:, 0:8]
Y = data_set[:, 8]

seed = 7
test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                    random_state=seed)

model = XGBClassifier()
model.fit(X_train, Y_train)

# save model to file
# pickle.dump(model, open("pima.pickle.dat", "wb"))
joblib.dump(model, "pima.joblib.dat")
print("save model to file")

# load model from file
# loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
loaded_model = joblib.load("pima.joblib.dat")
print("load model from file")

# y_pred = model.predict(X_test)
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)
print("accuracy: %.4f%%" % (accuracy * 100.0))
print("the model information: ", model)

# print("the feature importance: ", model.feature_importances_)
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)

print("the feature importance: ")
plot_importance(model)

plot_tree(model, num_trees=0, rankdir="LR")
plt.show()
















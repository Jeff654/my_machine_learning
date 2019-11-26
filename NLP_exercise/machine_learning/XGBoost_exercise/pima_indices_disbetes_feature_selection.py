# -*- coding: utf-8 -*-

from numpy import loadtxt, sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

data_set = loadtxt("XGBoost_resource/pima-indians-diabetes.csv", delimiter=",")

X = data_set[:, 0:8]
Y = data_set[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33,
                                                    random_state=7)

model = XGBClassifier()
# eval_set = [(X_test, Y_test)]
eval_set = [(X_train, Y_train), (X_test, Y_test)]

# model.fit(X_train, Y_train, eval_metric="error", eval_set=eval_set, verbose=True)
model.fit(X_train, Y_train, eval_metric=["error", "logloss"],
          eval_set=eval_set, verbose=True)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("accuracy: %.4f%%" % (accuracy * 100.0))



# model = XGBClassifier()
# model.fit(X_train, Y_train)
#
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(Y_test, predictions)
# print("accuracy: %.4f%%" % (accuracy * 100.0))


# fit model using each importance as a threshold
# thresholds = sort(model.feature_importances_)
# for thresh in thresholds:
#     # select features using thresh
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     selection_X_train = selection.transform(X_train)
#
#     # train model
#     selection_model = XGBClassifier()
#     selection_model.fit(selection_X_train, Y_train)
#
#     # eval model
#     selection_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(selection_X_test)
#
#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(Y_test, predictions)
#
#     print("thresh = %.4f, n = %d, accuracy: %.4f%%" % (thresh, selection_X_train.shape[1], accuracy * 100.0))


# 可视化部分
results = model.evals_result()
epochs = len(results["validation_0"]["error"])
x_axis = range(0, epochs)

# plot log loss
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x_axis, results["validation_0"]["logloss"], label="train")
ax.plot(x_axis, results["validation_1"]["logloss"], label="test")
ax.legend()
plt.ylabel("Log loss")
plt.title("XGBoost Log Loss")
plt.show()


# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results["validation_0"]["error"], label="train")
ax.plot(x_axis, results["validation_1"]["error"], label="test")
ax.legend()
plt.ylabel("Classification Error")
plt.title("XGBoost Classification Error")
plt.show()




















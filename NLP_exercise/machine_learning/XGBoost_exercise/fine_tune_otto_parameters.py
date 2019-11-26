# -*- coding: utf-8 -*-

from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("Agg")

data = read_csv("XGBoost_resource/otto-group-product-classification-challenge/train.csv")
data_set = data.values

X = data_set[:, 0:94]
Y = data_set[:, 94]

label_encoded_y = LabelEncoder().fit_transform(Y)

# grid search
model = XGBClassifier()
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss",
                           n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)


# summary results
print("best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]


for mean, std, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, std, param))


# plot
plt.errorbar(n_estimators, means, yerr=stds)
plt.title("XGBoost n_estimators VS Log Loss")
plt.xlabel("n_estimators")
plt.ylabel("Log Loss")
plt.savefig("n_estimators.png")

























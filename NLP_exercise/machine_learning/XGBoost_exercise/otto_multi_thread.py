# -*- coding: utf-8 -*-

from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time
import matplotlib.pyplot as plt

data = read_csv("XGBoost_resource/otto-group-product-classification-challenge/train.csv")
data_set = data.values

X = data_set[:, 0:94]
Y = data_set[:, 94]

label_encoded_y = LabelEncoder().fit_transform(Y)

#
# # evaluate the effect of the number of threads
# results = []
# number_threads = [1, 2, 3, 4]
# for n_core in number_threads:
#     start = time.time()
#     model = XGBClassifier(nthread=n_core)
#     model.fit(X, label_encoded_y)
#     elapsed = time.time() - start
#     print(n_core, elapsed)
#     results.append(elapsed)
#
# # plot
# plt.plot(number_threads, results)
# plt.ylabel("Speed (seconds)")
# plt.xlabel("Number of Threads")
# plt.title("XGBoost training speed VS Number of threads")
# plt.show()


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

# single thread XGBoost, Parallel thread CV
start = time.time()
model = XGBClassifier(nthread=1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold,
                          scoring="neg_log_loss", n_jobs=-1)
elapsed = time.time() - start
print("single thread XGBoost, parallel thread CV: %f" % elapsed)


# parallel thread XGBoost, single thread CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold,
                          scoring="neg_log_loss", n_jobs=1)
elapsed = time.time() - start
print("parallel thread XGboost, single thread CV: %f" % elapsed)


# parallel thread XGBoost and CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold,
                          scoring="neg_log_loss", n_jobs=-1)
elapsed = time.time() - start
print("parallel thread XGBoost and CV: %f" % elapsed)




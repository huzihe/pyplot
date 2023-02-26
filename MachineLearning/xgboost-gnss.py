"""
Author: huzihe06@gmail.com
Date: 2023-02-14 20:56:58
LastEditTime: 2023-02-25 18:07:57
FilePath:
"""


from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance

path = "./data/ml-data/gnss-data-20230129-1.csv"
gnssdata = pd.read_csv(path)
print(gnssdata.shape)
print(gnssdata.describe())
# print(gnssdata.isnull().any())
x = gnssdata.drop(["los", "prn"], axis=1)
y = gnssdata["los"]

# 2.划分数据与标签
train_data, test_data, train_label, test_label = train_test_split(
    x, y, random_state=1, train_size=0.6, test_size=0.4
)  # sklearn.model_selection.
# print(train_data.shape)

# # 3.训练svm分类器
# classifier = svm.SVC(
#     C=10, kernel="rbf", gamma=10, decision_function_shape="ovr"
# )  # ovr:一对多策略
# classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先


xgb_classifier = XGBClassifier(n_estimators=2000, eval_metric="logloss", eta=0.4)
# xgb_classifier.fit(train_data, train_label)
# define the datasets to evaluate each iteration
evalset = [(train_data, train_label), (test_data, test_label)]
# fit the model
xgb_classifier.fit(train_data, train_label, eval_set=evalset)

# 4.计算xgb分类器的准确率
print("训练集：", xgb_classifier.score(train_data, train_label))
print("测试集：", xgb_classifier.score(test_data, test_label))

# plot_importance(xgb_classifier)  # 绘制特征重要性
# plt.show()
# plt.savefig("./data/ml-data/gnss-xgboost-im.jpg")  # 保存图片

# evaluate performance
yhat = xgb_classifier.predict(test_data)
score = accuracy_score(test_label, yhat)
print("Accuracy: %.3f" % score)

results = xgb_classifier.evals_result()

# plot learning curves
plt.plot(results["validation_0"]["logloss"], label="train")
plt.plot(results["validation_1"]["logloss"], label="test")
# show the legend
plt.legend()
# show the plot
plt.savefig("./data/ml-data/mkrate-gnss-xgboost-2000.jpg")  # 保存图片

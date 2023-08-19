"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-13 20:20:36
LastEditTime: 2023-08-19 21:38:40
FilePath: /pyplot/MachineLearning/gnss_svm.py
Descripttion: 
"""

# -*- coding:utf-8 -*-

from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

from joblib import dump, load
from sklearn.metrics import accuracy_score


def gnss_svm_train_model(traindata, model):
    # model = "./data/ml-data/svmmodel.model"
    # path = "./data/ml-data/X6833B.res1"

    # 1.读取数据集
    gnssdata = pd.read_csv(traindata)
    x = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    y = gnssdata["los"]

    # 2.划分数据与标签
    train_data, test_data, train_label, test_label = train_test_split(
        x, y, random_state=1, train_size=0.6, test_size=0.4
    )  # sklearn.model_selection.

    # 3.训练svm分类器
    classifier = svm.SVC(
        C=2, kernel="rbf", gamma=10, decision_function_shape="ovo"
    )  # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

    dump(classifier, model)


def gnss_svm_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(39091)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    # X = Normalizer().fit_transform(originX)
    X = originX

    svm = load(model)
    labels = svm.predict(X)

    # evaluate performance
    score = accuracy_score(y, labels)
    print("Accuracy: %.3f" % score)

    satInfo.insert(loc=len(satInfo.columns), column="los", value=labels)
    return satInfo


if __name__ == "__main__":
    path1 = "./data/ml-data/trimble.res1"
    path2 = "./data/ml-data/X6833B.res1"
    modelpath1 = "./data/ml-data/gnss_svm_trimble.model"
    modelpath2 = "./data/ml-data/gnss_svm_X6833B.model"
    # gnss_svm_train_model(path1, modelpath1)
    # gnss_svm_train_model(path2, modelpath2)

    # satinfo_ref = gnss_svm_predict(modelpath1, path1)
    # satinfo_ref = gnss_svm_predict(modelpath2, path2)
    # satinfo_ref = gnss_svm_predict(modelpath1, path2)
    satinfo_ref = gnss_svm_predict(modelpath2, path1)


# # 4.计算svc分类器的准确率
# # print("训练集：", classifier.score(train_data, train_label))
# print("测试集：", svm.score(test_data, test_label))

# # 也可直接调用accuracy_score方法计算准确率


# # tra_label = classifier.predict(train_data)  # 训练集的预测标签
# tes_label = svm.predict(test_data)  # 测试集的预测标签
# # print("训练集：", accuracy_score(train_label, tra_label))
# print("测试集：", accuracy_score(test_label, tes_label))

# # # 查看决策函数
# # print("train_decision_function:\n", classifier.decision_function(train_data))  # (90,3)
# # print("predict_result:\n", classifier.predict(train_data))

# scores = []
# for m in range(2, 800):  # 循环2-79
#     classifier.fit(train_data[:m], train_label[:m])
#     y_train_predict = classifier.predict(train_data[:m])
#     y_val_predict = classifier.predict(test_data)
#     scores.append(accuracy_score(y_train_predict, train_label[:m]))
# plt.plot(range(2, 800), scores, c="green", alpha=0.6)
# plt.savefig("./data/ml-data/mkrate-gnss.jpg")  # 保存图片

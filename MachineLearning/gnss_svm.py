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
import time

from joblib import dump, load
from sklearn.metrics import accuracy_score


def gnss_svm_train_model(traindata, model):
    # model = "./data/ml-data/svmmodel.model"
    # path = "./data/ml-data/X6833B.res1"

    # 1.读取数据集
    gnssdata = pd.read_csv(traindata)
    x = gnssdata[["postResp", "priorR", "elevation", "SNR", "resSNR"]]
    y = gnssdata["los"]

    start  = time.time()
    # 2.划分数据与标签
    train_data, test_data, train_label, test_label = train_test_split(
        x, y, random_state=1, train_size=0.6, test_size=0.4
    )  # sklearn.model_selection.

    # 3.训练svm分类器
    classifier = svm.SVC(
        C=2, kernel="rbf", gamma=10, decision_function_shape="ovo"
    )  # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

    end  = time.time()
    print("预测算法耗时：",end - start)

    dump(classifier, model)


def gnss_svm_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(39091)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR", "resSNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    # X = Normalizer().fit_transform(originX)
    X = originX

    start  = time.time()
    svm = load(model)
    labels = svm.predict(X)

    end  = time.time()
    print("预测算法耗时：",end - start)

    # evaluate performance
    score = accuracy_score(y, labels)
    print("Accuracy: %.3f" % score)

    satInfo.insert(loc=len(satInfo.columns), column="los", value=labels)
    return satInfo


if __name__ == "__main__":
    trimble_path = "./data/ml-data/20230511/trimble.res1"
    X6833B_path = "./data/ml-data/20230511/X6833B.res1"
    ublox_path = "./data/ml-data/20230511/ublox.res1"
    CK6n_path = "./data/ml-data/20230511/CK6n.res1"

    path1 = "./data/ml-data/trimble.res1"
    path2 = "./data/ml-data/X6833B.res1"
    modelpath1 = "./data/ml-data/gnss_svm_trimble.model"
    modelpath2 = "./data/ml-data/gnss_svm_X6833B.model"
    modeltest = "./data/ml-data/gnss_svm_test.model"
    # gnss_svm_train_model(path1, modelpath1)
    # gnss_svm_train_model(path2, modelpath2)

    # gnss_svm_train_model(trimble_path, modeltest)
    # gnss_svm_train_model(ublox_path, modeltest)
    # gnss_svm_train_model(X6833B_path, modeltest)
    svm_trimble_modelpath = "./data/ml-data/model/gnss_svm_trimble.model"
    svm_X6833B_modelpath = "./data/ml-data/model/gnss_svm_X6833B.model"
    svm_ublox_modelpath = "./data/ml-data/model/gnss_svm_ublox.model"
    svm_CK6n_modelpath = "./data/ml-data/model/gnss_svm_CK6n.model"

    alloy_path = "./data/202401/log-spp-alloy.res1"
    ublox_path = "./data/202401/log-spp-ublox.res1"
    p40_path = "./data/202401/log-spp-p40.res1"
    alloy_modelpth = "./data/202401/model/gnss_svm_alloy.model"
    ublox_modelpth = "./data/202401/model/gnss_svm_ublox.model"
    p40_modelpth = "./data/202401/model/gnss_svm_p40.model"
    gnss_svm_train_model(alloy_path, alloy_modelpth)
    gnss_svm_train_model(ublox_path, ublox_modelpth)
    gnss_svm_train_model(p40_path, p40_modelpth)

    satinfo_ref = gnss_svm_predict(alloy_modelpth, alloy_path)
    satinfo_ref = gnss_svm_predict(alloy_modelpth, ublox_path)
    satinfo_ref = gnss_svm_predict(alloy_modelpth, p40_path)

    satinfo_ref = gnss_svm_predict(ublox_modelpth, alloy_path)
    satinfo_ref = gnss_svm_predict(ublox_modelpth, ublox_path)
    satinfo_ref = gnss_svm_predict(ublox_modelpth, p40_path)

    satinfo_ref = gnss_svm_predict(p40_modelpth, alloy_path)
    satinfo_ref = gnss_svm_predict(p40_modelpth, ublox_path)
    satinfo_ref = gnss_svm_predict(p40_modelpth, p40_path)

    # satinfo_ref = gnss_svm_predict(svm_trimble_modelpath, trimble_path)
    # satinfo_ref = gnss_svm_predict(svm_ublox_modelpath, ublox_path)
    # satinfo_ref = gnss_svm_predict(svm_X6833B_modelpath, X6833B_path)
    # satinfo_ref = gnss_svm_predict(modelpath2, path1)


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

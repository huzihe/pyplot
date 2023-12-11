"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-24 14:40:52
LastEditTime: 2023-08-19 20:33:21
FilePath: /pyplot/MachineLearning/gnss_GBDT.py
Descripttion: 
"""

from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
# from xgboost import Booster
# from xgboost import plot_importance
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from com.mytime import ymdhms2gpsws
from com.mystr import replace_char


def gbdt_gnss_train_model(traindata, model):
    gnssdata = pd.read_csv(traindata)
    print(gnssdata.shape)
    print(gnssdata.describe())
    # 数据分段处理，可选项
    # # 2023-5-11 对应 周内秒 2261 356400
    # X6833B  对应前39091行为静态数据，后12670（51761-39091）行为动态数据
    # trimble 对应前46082行为静态数据，后9873（55955-46082）行为动态数据
    # ublox 对应33405行为静态数据，后12978（46383-33405）行为动态数据
    # CK6n 对应33011行为静态数据，后9579（42590-33011）行为动态数据
    # gnssdata = gnssdata.head(33011)
    # # gnssdata = gnssdata.tail(9873)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    x = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    # week,second,sat,los,priorResp,postResp,priorR,postR,P,L,azimuth,elevation,SNR
    y = gnssdata["los"]

    # 2.划分数据与标签
    train_data, test_data, train_label, test_label = train_test_split(
        x, y, random_state=1, train_size=0.99, test_size=0.01
    )  # sklearn.model_selection.

    start  = time.time()

    gbdt_classifier = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)

    # # define the datasets to evaluate each iteration
    evalset = [(train_data, train_label), (test_data, test_label)]
    # fit the model
    gbdt_classifier.fit(train_data, train_label)
    # save model
    # gbdt_classifier.save_model(model)
    dump(gbdt_classifier, model)

    end  = time.time()
    print("预测算法耗时：",end - start)

    # 4.计算xgb分类器的准确率
    print("训练集：", gbdt_classifier.score(train_data, train_label))
    print("测试集：", gbdt_classifier.score(test_data, test_label))

    # evaluate performance
    yhat = gbdt_classifier.predict(test_data)
    score = accuracy_score(test_label, yhat)
    print("Accuracy: %.3f" % score)

    # plot_importance(gdbt_classifier)


def gbdt_gnss_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(33011)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    x = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    start  = time.time()

    gbdt_classifier = load(model)
    yhat = gbdt_classifier.predict(x)

    end  = time.time()
    print("预测算法耗时：",end - start)

    # evaluate performance
    score = accuracy_score(y, yhat)
    print("Accuracy: %.3f" % score)

    # 绘制特征重要性
    # plot_importance(xgb_classifier)
    

    satInfo.insert(loc=len(satInfo.columns), column="los", value=yhat)
    satInfo["second"] = satInfo["second"].astype(int)
    return satInfo


if __name__ == "__main__":
    trimble_path = "./data/ml-data/20230511/trimble.res1"
    X6833B_path = "./data/ml-data/20230511/X6833B.res1"
    ublox_path = "./data/ml-data/20230511/ublox.res1"
    CK6n_path = "./data/ml-data/20230511/CK6n.res1"

    trimble_modelpath = "./data/ml-data/model/gnss_gbdt_trimble-static.model"
    X6833B_modelpath = "./data/ml-data/model/gnss_gbdt_X6833B-static.model"
    ublox_modelpath = "./data/ml-data/model/gnss_gbdt_blox-static.model"
    CK6n_modelpath = "./data/ml-data/model/gnss_gbdt_CK6n-static.model"

    test_modelpath = "./data/ml-data/model/gnss_gbdt_test.model"

    # gbdt_gnss_train_model(trimble_path, trimble_modelpath)
    # gbdt_gnss_train_model(X6833B_path, X6833B_modelpath)
    # gbdt_gnss_train_model(ublox_path, ublox_modelpath)
    # gbdt_gnss_train_model(CK6n_path, CK6n_modelpath)
    gbdt_gnss_train_model(trimble_path, test_modelpath)
    gbdt_gnss_train_model(ublox_path, test_modelpath)
    gbdt_gnss_train_model(X6833B_path, test_modelpath)

    # satinfo_ref = gbdt_gnss_predict(CK6n_modelpath, trimble_path)
    # satinfo_ref = gbdt_gnss_predict(CK6n_modelpath, ublox_path)
    # satinfo_ref = gbdt_gnss_predict(CK6n_modelpath, X6833B_path)
    # satinfo_ref = gbdt_gnss_predict(CK6n_modelpath, CK6n_path)

    satinfo_ref = gbdt_gnss_predict(trimble_modelpath, trimble_path)
    # satinfo_ref = gbdt_gnss_predict(trimble_modelpath, ublox_path)
    # satinfo_ref = gbdt_gnss_predict(trimble_modelpath, X6833B_path)
    # satinfo_ref = gbdt_gnss_predict(trimble_modelpath, CK6n_path)

    # satinfo_ref = gbdt_gnss_predict(ublox_modelpath, trimble_path)
    satinfo_ref = gbdt_gnss_predict(ublox_modelpath, ublox_path)
    # satinfo_ref = gbdt_gnss_predict(ublox_modelpath, X6833B_path)
    # satinfo_ref = gbdt_gnss_predict(ublox_modelpath, CK6n_path)

    # satinfo_ref = gbdt_gnss_predict(X6833B_modelpath, trimble_path)
    # satinfo_ref = gbdt_gnss_predict(X6833B_modelpath, ublox_path)
    satinfo_ref = gbdt_gnss_predict(X6833B_modelpath, X6833B_path)
    # satinfo_ref = gbdt_gnss_predict(X6833B_modelpath, CK6n_path)



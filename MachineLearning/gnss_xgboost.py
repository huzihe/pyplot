"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-24 14:40:52
LastEditTime: 2023-08-19 20:33:21
FilePath: /pyplot/MachineLearning/gnss_xgboost.py
Descripttion: 
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
from xgboost import Booster
from xgboost import plot_importance
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from com.mytime import ymdhms2gpsws
from com.mystr import replace_char


def xgboost_gnss_train_model(traindata, model):
    gnssdata = pd.read_csv(traindata)
    # print(gnssdata.shape)
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
    x = gnssdata[["postResp", "elevation", "SNR", "resSNR"]]
    # week,second,sat,los,priorResp,postResp,priorR,postR,P,L,azimuth,elevation,SNR
    y = gnssdata["los"]

    start = time.time()

    # 2.划分数据与标签
    train_data, test_data, train_label, test_label = train_test_split(
        x, y, random_state=1, train_size=0.99, test_size=0.01
    )  # sklearn.model_selection.

    xgb_classifier = XGBClassifier(n_estimators=500, eval_metric="logloss", eta=0.3)

    # # define the datasets to evaluate each iteration
    evalset = [(train_data, train_label), (test_data, test_label)]
    # fit the model
    xgb_classifier.fit(train_data, train_label, eval_set=evalset)

    end  = time.time()
    print("训练算法耗时：",end - start)
    # save model
    xgb_classifier.save_model(model)

    # 4.计算xgb分类器的准确率
    print("训练集：", xgb_classifier.score(train_data, train_label))
    print("测试集：", xgb_classifier.score(test_data, test_label))

    # evaluate performance
    yhat = xgb_classifier.predict(test_data)
    score = accuracy_score(test_label, yhat)
    print("Accuracy: %.3f" % score)

    plot_importance(xgb_classifier)


def xgboost_gnss_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(33011)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    x = gnssdata[["postResp", "elevation", "SNR", "resSNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    start  = time.time()

    xgb_classifier = XGBClassifier()
    booster = Booster()
    booster.load_model(model)
    xgb_classifier._Booster = booster

    # evaluate performance
    yhat = xgb_classifier.predict(x)

    end  = time.time()
    print("预测算法耗时：",end - start)

    # score = accuracy_score(y, yhat)
    # print("Accuracy: %.3f" % score)

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

    modelpath1 = "./data/ml-data/model/gnss_xgboost_trimble.model"
    modelpath2 = "./data/ml-data/model/gnss_xgboost_X6833B.model"
    modeltest = "./data/ml-data/model/gnss_xgboost_X6833B-test.model"

    trimble_modelpath = "./data/ml-data/model/gnss_xgboost_trimble-static.model"
    X6833B_modelpath = "./data/ml-data/model/gnss_xgboost_X6833B-static.model"
    ublox_modelpath = "./data/ml-data/model/gnss_xgboost_blox-static.model"
    CK6n_modelpath = "./data/ml-data/model/gnss_xgboost_CK6n-static.model"

    # all_path = "./data/202401/log-spp-all.res1"
    alloy_path = "./data/202401/log-spp-alloy.res1"
    ublox_path = "./data/202401/log-spp-ublox.res1"
    p40_path = "./data/202401/log-spp-p40.res1"
    # all_modelpth = "./data/202401/model/gnss_xgboost_all.model"
    alloy_modelpth = "./data/202401/model/gnss_xgboost_alloy.model"
    ublox_modelpth = "./data/202401/model/gnss_xgboost_ublox.model"
    p40_modelpth = "./data/202401/model/gnss_xgboost_p40.model"
    # alloy_path = "./data/202401/log-spp-alloy-0120-train.res1"
    # ublox_path = "./data/202401/log-spp-ublox-0120-train.res1"
    # p40_path = "./data/202401/log-spp-p40-0120-train.res1"
    # alloy_modelpth = "./data/202401/model/gnss_xgboost_alloy0120.model"
    # ublox_modelpth = "./data/202401/model/gnss_xgboost_ublox0120.model"
    # p40_modelpth = "./data/202401/model/gnss_xgboost_p400120.model"
    # xgboost_gnss_train_model(all_path, all_modelpth)
    xgboost_gnss_train_model(alloy_path, alloy_modelpth)
    xgboost_gnss_train_model(ublox_path, ublox_modelpth)
    xgboost_gnss_train_model(p40_path, p40_modelpth)

    # satinfo_ref = xgboost_gnss_predict(all_modelpth, alloy_path)
    # satinfo_ref = xgboost_gnss_predict(all_modelpth, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(all_modelpth, p40_path)

    # satinfo_ref = xgboost_gnss_predict(alloy_modelpth, alloy_path)
    # satinfo_ref = xgboost_gnss_predict(alloy_modelpth, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(alloy_modelpth, p40_path)

    # satinfo_ref = xgboost_gnss_predict(ublox_modelpth, alloy_path)
    # satinfo_ref = xgboost_gnss_predict(ublox_modelpth, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(ublox_modelpth, p40_path)

    # satinfo_ref = xgboost_gnss_predict(p40_modelpth, alloy_path)
    # satinfo_ref = xgboost_gnss_predict(p40_modelpth, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(p40_modelpth, p40_path)

    # xgboost_gnss_train_model(trimble_path, modeltest)
    # xgboost_gnss_train_model(ublox_path, modeltest)
    # xgboost_gnss_train_model(X6833B_path, modeltest)
    # xgboost_gnss_train_model(trimble_path, trimble_modelpath)
    # xgboost_gnss_train_model(X6833B_path, X6833B_modelpath)
    # xgboost_gnss_train_model(ublox_path, ublox_modelpath)
    # xgboost_gnss_train_model(CK6n_path, CK6n_modelpath)

    # satinfo_ref = xgboost_gnss_predict(CK6n_modelpath, trimble_path)
    # satinfo_ref = xgboost_gnss_predict(CK6n_modelpath, X6833B_path)
    # satinfo_ref = xgboost_gnss_predict(CK6n_modelpath, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(CK6n_modelpath, CK6n_path)

    # satinfo_ref = xgboost_gnss_predict(trimble_modelpath, trimble_path)
    # satinfo_ref = xgboost_gnss_predict(trimble_modelpath, X6833B_path)
    # satinfo_ref = xgboost_gnss_predict(trimble_modelpath, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(trimble_modelpath, CK6n_path)

    # satinfo_ref = xgboost_gnss_predict(X6833B_modelpath, trimble_path)
    # satinfo_ref = xgboost_gnss_predict(X6833B_modelpath, X6833B_path)
    # satinfo_ref = xgboost_gnss_predict(X6833B_modelpath, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(X6833B_modelpath, CK6n_path)

    # satinfo_ref = xgboost_gnss_predict(ublox_modelpath, trimble_path)
    # satinfo_ref = xgboost_gnss_predict(ublox_modelpath, X6833B_path)
    # satinfo_ref = xgboost_gnss_predict(ublox_modelpath, ublox_path)
    # satinfo_ref = xgboost_gnss_predict(ublox_modelpath, CK6n_path)

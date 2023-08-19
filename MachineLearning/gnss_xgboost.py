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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from com.mytime import ymdhms2gpsws
from com.mystr import replace_char


def xgboost_gnss_train_model(traindata, model):
    gnssdata = pd.read_csv(traindata)
    print(gnssdata.shape)
    print(gnssdata.describe())
    # 数据分段处理，可选项
    # # 2023-5-11 对应 周内秒 2261 356400
    # # X6833B  对应前39091行为静态数据，后12670（51761-39091）行为动态数据
    # # trimble 对应前46082行为静态数据，后9873（55955-46082）行为动态数据
    # # gnssdata = gnssdata.head(46082)
    # # gnssdata = gnssdata.tail(9873)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    x = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    # week,second,sat,los,priorResp,postResp,priorR,postR,P,L,azimuth,elevation,SNR
    y = gnssdata["los"]

    # 2.划分数据与标签
    train_data, test_data, train_label, test_label = train_test_split(
        x, y, random_state=1, train_size=0.5, test_size=0.5
    )  # sklearn.model_selection.

    xgb_classifier = XGBClassifier(n_estimators=200, eval_metric="logloss", eta=0.3)

    # # define the datasets to evaluate each iteration
    evalset = [(train_data, train_label), (test_data, test_label)]
    # fit the model
    xgb_classifier.fit(train_data, train_label, eval_set=evalset)
    # save model
    xgb_classifier.save_model(model)

    # 4.计算xgb分类器的准确率
    print("训练集：", xgb_classifier.score(train_data, train_label))
    print("测试集：", xgb_classifier.score(test_data, test_label))

    # evaluate performance
    yhat = xgb_classifier.predict(test_data)
    score = accuracy_score(test_label, yhat)
    print("Accuracy: %.3f" % score)


def xgboost_gnss_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    x = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    xgb_classifier = XGBClassifier()
    booster = Booster()
    booster.load_model(model)
    xgb_classifier._Booster = booster

    # evaluate performance
    yhat = xgb_classifier.predict(x)
    score = accuracy_score(y, yhat)
    print("Accuracy: %.3f" % score)

    satInfo.insert(loc=len(satInfo.columns), column="los", value=yhat)
    return satInfo


# def out_gnss_data(rnx, outrnx, satInfo, istrimble):
#     result = {}
#     with open(rnx, "r") as file:
#         content = file.readlines()
#         headerflag = False
#         with open(outrnx, "w") as outfile:
#             for eachLine in content:
#                 if eachLine.startswith("%"):  # ignore comment line
#                     outfile.write(eachLine)  # 输出 rnx 文件头
#                     continue
#                 ender = "END OF HEADER" in eachLine
#                 if ender:
#                     headerflag = True
#                     outfile.write(eachLine)  # 输出 rnx 文件头
#                     continue
#                 elif headerflag:
#                     if eachLine.startswith(">"):
#                         outfile.write(eachLine)  # 输出时间部分

#                         eachData = eachLine.split()
#                         ws = ymdhms2gpsws(
#                             int(eachData[1]),
#                             int(eachData[2]),
#                             int(eachData[3]),
#                             int(eachData[4]),
#                             int(eachData[5]),
#                             int(float(eachData[6])),
#                         )
#                         continue
#                     else:
#                         eachData = eachLine.split()
#                         los = satInfo.loc[
#                             (satInfo["week"] == ws[0])
#                             & (satInfo["second"] == ws[1])
#                             & (satInfo["sat"] == eachData[0]),
#                             :,
#                         ]
#                         if los.size <= 0:
#                             outfile.write(eachLine)
#                         else:
#                             nlosflag = str(los.iat[0, 3])
#                             for itr in range(6):
#                                 if len(eachLine) > 48 * itr + 18 + 16:
#                                     substr = eachLine[48 * itr + 18]
#                                     if substr != " ":
#                                         eachLine = replace_char(
#                                             eachLine, nlosflag, 48 * itr + 18
#                                         )
#                                         eachLine = replace_char(
#                                             eachLine, nlosflag, 48 * itr + 18 + 16
#                                         )
#                                         if not istrimble:
#                                             eachLine = replace_char(
#                                                 eachLine, nlosflag, 48 * itr + 18 + 32
#                                             )
#                                             eachLine = replace_char(
#                                                 eachLine, nlosflag, 48 * itr + 18 + 48
#                                             )
#                             outfile.write(eachLine)  # 输出nlos修订后的obs记录
#                         continue
#                 else:
#                     outfile.write(eachLine)  # 输出 rnx 文件头
#                     continue


if __name__ == "__main__":
    path1 = "./data/ml-data/trimble.res1"
    path2 = "./data/ml-data/X6833B.res1"
    modelpath1 = "./data/ml-data/gnss_xgboost_trimble.model"
    modelpath2 = "./data/ml-data/gnss_xgboost_X6833B.model"
    # xgboost_gnss_train_model(path1, modelpath1)
    # xgboost_gnss_train_model(path2, modelpath2)

    # satinfo_ref = xgboost_gnss_predict(modelpath1, path1)
    # satinfo_ref = xgboost_gnss_predict(modelpath2, path2)
    # satinfo_ref = xgboost_gnss_predict(modelpath1, path2)
    satinfo_ref = xgboost_gnss_predict(modelpath2, path1)

    # rnx = "./data/ml-data/trimble-3dma-0520.rnx"
    # outrnx = "./data/ml-data/trimble-3dma-0520-ai.rnx"
    # out_gnss_data(rnx, outrnx, satinfo_ref, 1)
    rnx = "./data/ml-data/X6833B-3dma-0730.rnx"
    outrnx = "./data/ml-data/X6833B-3dma-0730-ai.rnx"
    # out_gnss_data(rnx, outrnx, satinfo_ref, 0)

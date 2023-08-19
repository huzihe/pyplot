"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-08-19 20:38:37
LastEditTime: 2023-08-19 20:39:12
FilePath: /pyplot/MachineLearning/gnss_fcm.py
Descripttion: 
"""

import numpy as np
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
from sklearn import metrics

from joblib import dump, load


def gnss_fcm_train_model(traindata, model):
    gnssdata = pd.read_csv(traindata)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    y = gnssdata["los"]

    # 数据标准化
    X = Normalizer().fit_transform(originX)

    # 聚类
    fcm = FCM()
    fcm.n_clusters = 2
    fcm.fit(X)

    dump(fcm, model)


def gnss_fcm_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    X = Normalizer().fit_transform(originX)

    fcm = load(model)
    labels = fcm.predict(X)
    # labels = 1 - labels  # 将预测结果和 nlos实际意义做对应调整

    sil_samples = metrics.silhouette_samples(X, labels)

    from sklearn.metrics import accuracy_score

    score = accuracy_score(y, labels)
    print(
        "fcm silhouette score = {:.3}, accuracy score = {:.3}\n".format(
            sil_samples.mean(), score
        )
    )

    satInfo.insert(loc=len(satInfo.columns), column="los", value=labels)
    return satInfo


if __name__ == "__main__":
    path1 = "./data/ml-data/trimble.res1"
    path2 = "./data/ml-data/X6833B.res1"
    modelpath1 = "./data/ml-data/gnss_fcm_trimble.model"
    modelpath2 = "./data/ml-data/gnss_fcm_X6833B.model"
    # gnss_fcm_train_model(path1, modelpath1)
    # gnss_fcm_train_model(path2, modelpath2)

    satinfo_ref = gnss_fcm_predict(modelpath2, path1)
    # satinfo_ref2 = gnss_fcm_predict(modelpath2, path2)

'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-08-19 20:38:37
LastEditTime: 2024-02-16 09:51:49
FilePath: /pyplot/MachineLearning/gnss_fcm.py
Descripttion: 
'''
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
import time


def gnss_fcm_train_model(traindata, model):
    gnssdata = pd.read_csv(traindata)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    y = gnssdata["los"]

    start = time.time()
    # 数据标准化
    X = Normalizer().fit_transform(originX)

    # 聚类
    fcm = FCM()
    fcm.n_clusters = 2
    fcm.fit(X)

    dump(fcm, model)
    end = time.time()
    print("fcm 训练算法耗时：",end - start)


def gnss_fcm_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(5000)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    start = time.time()
    X = Normalizer().fit_transform(originX)

    fcm = load(model)
    labels = fcm.predict(X)
    labels = 1 - labels  # 将预测结果和 nlos实际意义做对应调整

    end = time.time()
    print("fcm 预测算法耗时：",end - start)

    from sklearn.metrics import accuracy_score
    score = accuracy_score(y, labels)
    print("fcm accuracy score = {:.3}\n".format(score))

    sil_samples = metrics.silhouette_samples(X, labels)
    score = accuracy_score(y, labels)
    print(
        "fcm silhouette score = {:.3}, accuracy score = {:.3}\n".format(
            sil_samples.mean(), score
        )
    )

    satInfo.insert(loc=len(satInfo.columns), column="los", value=labels)
    return satInfo


if __name__ == "__main__":
    # trimble_path = "./data/ml-data/20230511/trimble.res1"
    # X6833B_path = "./data/ml-data/20230511/X6833B.res1"
    # ublox_path = "./data/ml-data/20230511/ublox.res1"
    # CK6n_path = "./data/ml-data/20230511/CK6n.res1"

    trimble_modelpath = "./data/ml-data/model/gnss_fcm_trimble.model"
    X6833B_modelpath = "./data/ml-data/model/gnss_fcm_X6833B.model"
    ublox_modelpath = "./data/ml-data/model/gnss_fcm_ublox.model"
    CK6n_modelpath = "./data/ml-data/model/gnss_fcm_CK6n.model"

    # gnss_fcm_train_model(trimble_path, trimble_modelpath)
    # gnss_fcm_train_model(X6833B_path, X6833B_modelpath)
    # gnss_fcm_train_model(ublox_path, ublox_modelpath)
    # # gnss_fcm_train_model(CK6n_path, CK6n_modelpath)

    # satinfo = gnss_fcm_predict(trimble_modelpath, trimble_path)
    # satinfo = gnss_fcm_predict(X6833B_modelpath, X6833B_path)
    # satinfo = gnss_fcm_predict(ublox_modelpath, ublox_path)
    # # satinfo = gnss_fcm_predict(CK6n_modelpath, CK6n_path)

    # 202401 data 
    alloy_path = "./data/202401/log-spp-alloy-new.res1"
    ublox_path = "./data/202401/log-spp-ublox-new.res1"
    p40_path = "./data/202401/log-spp-p40-new.res1"
    alloy_modelpth = "./data/202401/model/gnss_fcm_alloy.model"
    ublox_modelpth = "./data/202401/model/gnss_fcm_ublox.model"
    p40_modelpth = "./data/202401/model/gnss_fcm_p40.model"
    # gnss_fcm_train_model(alloy_path, alloy_modelpth)
    # gnss_fcm_train_model(ublox_path, ublox_modelpth)
    # gnss_fcm_train_model(p40_path, p40_modelpth)

    satinfo_ref = gnss_fcm_predict(alloy_modelpth, alloy_path)
    # satinfo_ref = gnss_fcm_predict(alloy_modelpth, ublox_path)
    # satinfo_ref = gnss_fcm_predict(alloy_modelpth, p40_path)

    # satinfo_ref = gnss_fcm_predict(ublox_modelpth, alloy_path)
    satinfo_ref = gnss_fcm_predict(ublox_modelpth, ublox_path)
    # satinfo_ref = gnss_fcm_predict(ublox_modelpth, p40_path)

    # satinfo_ref = gnss_fcm_predict(p40_modelpth, alloy_path)
    # satinfo_ref = gnss_fcm_predict(p40_modelpth, ublox_path)
    satinfo_ref = gnss_fcm_predict(p40_modelpth, p40_path)
    # 202401 data end

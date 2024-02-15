"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-18 15:56:36
LastEditTime: 2023-08-19 20:34:08
FilePath: /pyplot/MachineLearning/gnss_gmm.py
Descripttion: 
"""

import numpy as np
import time

# from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
from sklearn import metrics


def gnss_gmm_train_model(traindata, model):
    gnssdata = pd.read_csv(traindata)
    gnssdata = gnssdata.head(10000)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    y = gnssdata["los"]

    # 数据标准化
    X = Normalizer().fit_transform(originX)

    start = time.time()
    # for n_clusters in range(2, 5):
    train_data, test_data, train_label, test_label = train_test_split(
        X, y, random_state=1, train_size=0.7, test_size=0.3
    )

    # 聚类
    gmm = GaussianMixture(2, covariance_type="diag", random_state=0).fit(X)

    dump(gmm, model)

    end = time.time()
    print("gmm 训练算法耗时：",end - start)


def gnss_gmm_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(39091)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    X = Normalizer().fit_transform(originX)

    start = time.time()
    gmm = load(model)
    labels = gmm.predict(X)
    labels = 1 - labels  # 将预测结果和 nlos实际意义做对应调整

    end = time.time()
    print("gmm 预测算法耗时：",end - start)

    from sklearn.metrics import accuracy_score
    score = accuracy_score(y, labels)
    print("gmm accuracy score = {:.3}\n".format(score))

    sil_samples = metrics.silhouette_samples(X, labels)
    print(
        "gmm silhouette score = {:.3}, accuracy score = {:.3}\n".format(
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

    trimble_modelpath = "./data/ml-data/model/gnss_gmm_trimble.model"
    X6833B_modelpath = "./data/ml-data/model/gnss_gmm_X6833B.model"
    ublox_modelpath = "./data/ml-data/model/gnss_gmm_ublox.model"
    CK6n_modelpath = "./data/ml-data/model/gnss_gmm_CK6n.model"

    # gnss_gmm_train_model(trimble_path, trimble_modelpath)
    # gnss_gmm_train_model(X6833B_path, X6833B_modelpath)
    # gnss_gmm_train_model(ublox_path, ublox_modelpath)
    # gnss_gmm_train_model(CK6n_path, CK6n_modelpath)

    # satinfo = gnss_gmm_predict(trimble_modelpath, trimble_path)
    # satinfo = gnss_gmm_predict(X6833B_modelpath, X6833B_path)
    # satinfo = gnss_gmm_predict(ublox_modelpath, ublox_path)
    # satinfo = gnss_gmm_predict(CK6n_modelpath, CK6n_path)

    # 202401 data 
    # alloy_path = "./data/202401/log-spp-alloy.res1"
    # ublox_path = "./data/202401/log-spp-ublox.res1"
    # p40_path = "./data/202401/log-spp-p40.res1"
    alloy_path = "./data/202401/log-spp-alloy-new.res1"
    ublox_path = "./data/202401/log-spp-ublox-new.res1"
    p40_path = "./data/202401/log-spp-p40-new.res1"
    alloy_modelpath = "./data/202401/model/gnss_gmm_alloy.model"
    # ublox_modelpath = "./data/202401/model/gnss_gmm_ublox.model"
    p40_modelpath = "./data/202401/model/gnss_gmm_p40.model"
    # gnss_gmm_train_model(alloy_path, alloy_modelpath)
    # gnss_gmm_train_model(ublox_path, ublox_modelpath)
    # gnss_gmm_train_model(p40_path, p40_modelpath)

    # satinfo_ref = gnss_gmm_predict(alloy_modelpath, alloy_path)
    # satinfo_ref = gnss_gmm_predict(alloy_modelpath, ublox_path)
    # satinfo_ref = gnss_gmm_predict(alloy_modelpath, p40_path)

    satinfo_ref = gnss_gmm_predict(CK6n_modelpath, alloy_path)
    satinfo_ref = gnss_gmm_predict(CK6n_modelpath, ublox_path)
    satinfo_ref = gnss_gmm_predict(CK6n_modelpath, p40_path)

    # satinfo_ref = gnss_gmm_predict(p40_modelpath, alloy_path)
    # satinfo_ref = gnss_gmm_predict(p40_modelpath, ublox_path)
    # satinfo_ref = gnss_gmm_predict(p40_modelpath, p40_path)
    # 202401 data end

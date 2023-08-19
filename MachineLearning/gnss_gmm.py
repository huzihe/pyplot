"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-18 15:56:36
LastEditTime: 2023-08-19 20:34:08
FilePath: /pyplot/MachineLearning/gnss_gmm.py
Descripttion: 
"""

import numpy as np

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
    # gnssdata = gnssdata.head(39091)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    y = gnssdata["los"]

    # 数据标准化
    X = Normalizer().fit_transform(originX)

    # for n_clusters in range(2, 5):
    train_data, test_data, train_label, test_label = train_test_split(
        X, y, random_state=1, train_size=0.7, test_size=0.3
    )

    # 聚类
    gmm = GaussianMixture(2, covariance_type="diag", random_state=0).fit(X)

    dump(gmm, model)


def gnss_gmm_predict(model, testdata):
    gnssdata = pd.read_csv(testdata)
    # gnssdata = gnssdata.head(39091)
    # x = gnssdata.drop(["week","second","sat","los","priorResp","postR","P","L","azimuth",],axis=1,)
    originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
    satInfo = gnssdata[["week", "second", "sat"]]
    y = gnssdata["los"]

    X = Normalizer().fit_transform(originX)

    gmm = load(model)
    labels = gmm.predict(X)
    labels = 1 - labels  # 将预测结果和 nlos实际意义做对应调整

    sil_samples = metrics.silhouette_samples(X, labels)

    from sklearn.metrics import accuracy_score

    score = accuracy_score(y, labels)
    print(
        "gmm silhouette score = {:.3}, accuracy score = {:.3}\n".format(
            sil_samples.mean(), score
        )
    )

    satInfo.insert(loc=len(satInfo.columns), column="los", value=labels)
    return satInfo


if __name__ == "__main__":
    path1 = "./data/ml-data/trimble.res1"
    path2 = "./data/ml-data/X6833B.res1"
    modelpath1 = "./data/ml-data/gnss_gmm_trimble.model"
    modelpath2 = "./data/ml-data/gnss_gmm_X6833B.model"
    # gnss_gmm_train_model(path1, modelpath1)
    gnss_gmm_train_model(path2, modelpath2)

    # satinfo_ref = gnss_gmm_predict(modelpath1, path1)
    # satinfo_ref = gnss_gmm_predict(modelpath2, path1)
    satinfo_ref2 = gnss_gmm_predict(modelpath2, path2)

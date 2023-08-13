"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-18 15:56:36
LastEditTime: 2023-06-23 14:47:28
FilePath: /pyplot/MachineLearning/unclass-gmm.py
Descripttion: 
"""
"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-06-18 15:56:36
LastEditTime: 2023-06-18 15:59:45
FilePath: /pyplot/MachineLearning/unclass-gmm.py
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


import pandas as pd
from sklearn import metrics

# path = "./data/ml-data/gnss-data-20230129-1.csv"
# gnssdata = pd.read_csv(path)
# print(gnssdata.shape)
# print(gnssdata.describe())
# #print(gnssdata.isnull().any())
# originX = gnssdata.drop(["los", "prn"], axis=1)
# y = gnssdata["los"]

path = "./data/ml-data/X6833B.csv"
gnssdata = pd.read_csv(path)
print(gnssdata.shape)
print(gnssdata.describe())
originX = gnssdata[["postResp", "priorR", "elevation", "SNR"]]
# week,second,sat,los,priorResp,postResp,priorR,postR,P,L,azimuth,elevation,SNR
y = gnssdata["los"]

# 数据标准化
X = Normalizer().fit_transform(originX)

for n_clusters in range(2, 5):
    train_data, test_data, train_label, test_label = train_test_split(
        X, y, random_state=1, train_size=0.6, test_size=0.4
    )

    # 聚类
    gmm = GaussianMixture(n_clusters, covariance_type="diag", random_state=0).fit(X)

    X = test_data
    y = test_label

    labels = gmm.predict(X)
    sil_samples = metrics.silhouette_samples(X, labels)

    from sklearn.metrics import accuracy_score

    s = accuracy_score(y, labels)
    print(
        "for n_clusters = {}, silhouette score = {}, accuracy score = {}\n".format(
            n_clusters, sil_samples.mean(), s
        )
    )

    # 建立画布
    fig, charts = plt.subplots(1, 2)
    fig.set_size_inches(14, 5)

    interval = 20
    lower = 0
    higher = 0

    for i in range(n_clusters):
        sil_samples_i = sil_samples[labels == i]
        sil_samples_i.sort()
        higher = sil_samples_i.shape[0] + lower

        # 填充
        charts[0].fill_betweenx(
            np.arange(lower, higher),
            sil_samples_i,
            facecolor=cm.nipy_spectral(i / n_clusters),
            alpha=0.7,
        )

        # 显示类别
        charts[0].text(-0.05, (lower + higher) * 0.5, str(i))

        lower = higher + interval

    # 画出轮廓系数的均值线
    charts[0].axvline(x=sil_samples.mean(), color="red", linestyle="--")

    # 设置坐标轴
    charts[0].set_xlabel("silhouette scores")
    charts[0].set_ylabel("clusters={}".format(n_clusters))
    # charts[0].set_xticks(np.arange(-0.2, 1.2, 0.2))
    # charts[0].set_yticks([])

    # 画出聚类的结果散点图
    charts[1].scatter(X[:, 0], X[:, 1], c=labels)

    # 画出质心
    centers = gmm.means_
    charts[1].scatter(centers[:, 0], centers[:, 1], color="red", marker="x", s=80)

    plt.show()

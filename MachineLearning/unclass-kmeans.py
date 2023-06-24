"""
Author: hzh huzihe@whu.edu.cn
Date: 2022-12-08 17:28:22
LastEditTime: 2023-06-17 19:43:08
FilePath: /pyplot/MachineLearning/UnsupervisedClassification.py
Descripttion: 
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
from sklearn import metrics

# 加载数据
# data = load_iris()
# X = data["data"]
# y = data["target"]

path = "./data/ml-data/gnss-data-20230129-1.csv"
gnssdata = pd.read_csv(path)
# print(gnssdata.shape)
# print(gnssdata.describe())
originX = gnssdata.drop(["los", "prn"], axis=1)
y = gnssdata["los"]

# 数据标准化
X = Normalizer().fit_transform(originX)

for n_clusters in range(2, 5):
    # 聚类
    kmeans = KMeans(n_clusters).fit(X)
    labels = kmeans.labels_
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
    centers = kmeans.cluster_centers_
    charts[1].scatter(centers[:, 0], centers[:, 1], color="red", marker="x", s=80)

    plt.show()
    # print(
    #     "for n_clusters = {}, silhouette score = {}\n".format(
    #         n_clusters, sil_samples.mean()
    #     )
    # )

# s = metrics.silhouette_score(X, labels)
# print("轮廓系数:")
# print(str(s))


# x_axis = list(range(0, y.size))
# # plt.scatter(x_axis, samples)
# plt.plot_date(x_axis, samples, "c.")
# plt.show()


# from sklearn.metrics import accuracy_score

# print(labels)
# print(accuracy_score(y, labels))


# # 绘制聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()

# print(__doc__)

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs

# plt.figure(figsize=(8, 6))

# n_samples = 1500
# random_state = 170
# X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# # Incorrect number of clusters
# y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.title("Incorrect Number of Blobs")

# # Anisotropicly distributed data
# transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
# X_aniso = np.dot(X, transformation)
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

# plt.subplot(222)
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
# plt.title("Anisotropicly Distributed Blobs")

# # Different variance
# X_varied, y_varied = make_blobs(
#     n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
# )
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

# plt.subplot(223)
# plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
# plt.title("Unequal Variance")

# # Unevenly sized blobs
# X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)

# plt.subplot(224)
# plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
# plt.title("Unevenly Sized Blobs")

# plt.show()

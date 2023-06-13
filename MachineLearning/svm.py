"""
Author: Zihe Hu huzihe06@gmail.com
Date: 2022-12-25 11:26:03
LastEditors: Zihe Hu huzihe06@gmail.com
LastEditTime: 2022-12-26 20:59:11
FilePath: \pyplot\MachineLearning\svm.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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

# define converts(字典)
# def Iris_label(s):
#     it = {b"Iris-setosa": 0, b"Iris-versicolor": 1, b"Iris-virginica": 2}
#     return it[s]
def Calstr(s):
    return 1


# 1.读取数据集
# # path = "./data/ml-data/gnss-nlos-test.txt"
# path = "./data/ml-data/gnss-data-202210071.txt"
# data = np.loadtxt(path, dtype=float, delimiter=",")
# print(data.shape)

path = "./data/ml-data/gnss-data-20230129-1.csv"
gnssdata = pd.read_csv(path)
print(gnssdata.shape)
print(gnssdata.describe())
# print(gnssdata.isnull().any())
x = gnssdata.drop(['los','resp'], axis=1)
y = gnssdata['los']


# 2.划分数据与标签
train_data, test_data, train_label, test_label = train_test_split(
    x, y, random_state=1, train_size=0.6, test_size=0.4
)  # sklearn.model_selection.
# print(train_data.shape)

# 3.训练svm分类器
classifier = svm.SVC(
    C=2, kernel="rbf", gamma=10, decision_function_shape="ovo"
)  # ovr:一对多策略
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))

# 也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score

# tra_label = classifier.predict(train_data)  # 训练集的预测标签
# tes_label = classifier.predict(test_data)  # 测试集的预测标签
# print("训练集：", accuracy_score(train_label, tra_label))
# print("测试集：", accuracy_score(test_label, tes_label))

# 查看决策函数
print("train_decision_function:\n", classifier.decision_function(train_data))  # (90,3)
print("predict_result:\n", classifier.predict(train_data))

scores = []
for m in range(2,800):#循环2-79
    classifier.fit(train_data[:m],train_label[:m])
    y_train_predict = classifier.predict(train_data[:m])
    y_val_predict = classifier.predict(test_data)
    scores.append(accuracy_score(y_train_predict,train_label[:m]))
plt.plot(range(2,800),scores,c='green', alpha=0.6)
plt.savefig('./data/ml-data/mkrate-gnss.jpg')   # 保存图片

# # 5.绘制图形
# # 确定坐标轴范围
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
# grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# # 指定默认字体
# matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
# # 设置颜色
# cm_light = matplotlib.colors.ListedColormap(["#A0FFA0", "#FFA0A0", "#A0A0FF"])
# cm_dark = matplotlib.colors.ListedColormap(["g", "r", "b"])

# grid_hat = classifier.predict(grid_test)  # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
# # plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  # 样本
# plt.scatter(
#     test_data[:, 0], 
#     test_data[:, 1],
#     c=test_label[:, 0],
#     s=30,
#     edgecolors="k",
#     zorder=2,
#     cmap=cm_dark,
# )  # 圈中测试集样本点
# plt.xlabel("花萼长度", fontsize=13)
# plt.ylabel("花萼宽度", fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title("鸢尾花SVM二特征分类")
# plt.show()

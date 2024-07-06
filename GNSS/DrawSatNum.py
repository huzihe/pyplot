'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-10-29 21:36:51
LastEditTime: 2024-03-03 21:59:39
FilePath: /pyplot/GNSS/DrawSatNum.py
Descripttion: 
'''
# -*- coding:utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from com.readrnx import readrnx
from com.readrnx import StatisticResult
from com.mytime import gpsws2ymdhms
from datetime import datetime

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8


def DrawFigures(_stat,_figname):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 绘制散点图或折线图
    input      : 类cStat的对象:_stat
    """
    # 时间格式：%Y-%m-%d %H:%M:%S，x轴显示：%H:%M:%S
    ymdhms = np.zeros((len(_stat.gpsw), 6), dtype=float)
    Time_hms = []
    for ieph in range(0, len(_stat.gpsw)):
        (
            ymdhms[ieph][0],
            ymdhms[ieph][1],
            ymdhms[ieph][2],
            ymdhms[ieph][3],
            ymdhms[ieph][4],
            ymdhms[ieph][5],
        ) = gpsws2ymdhms(
            int(_stat.gpsw[ieph]),
            (_stat.gpsw[ieph] - int(_stat.gpsw[ieph])) * 86400 * 7,
        )
        Time_hms.append(
            "%04d-%02d-%02d %02d:%02d:%02d"
            % (
                ymdhms[ieph][0],
                ymdhms[ieph][1],
                ymdhms[ieph][2],
                ymdhms[ieph][3],
                ymdhms[ieph][4],
                ymdhms[ieph][5],
            )
        )
    Time_hms = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in Time_hms]

    limMax = 40
    inch = 1/2.54
    ## draw position
    plt.figure(dpi=300, figsize=(8.4*inch, 5*inch))
    myFmt = mdates.DateFormatter("%H:%M:%S")
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dx,
        color='C1',
        linestyle='',
        marker='.',
        markersize='2',
        label="satellite"
        )
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dy,
        color='C2',
        linestyle='',
        marker='.',
        markersize='2',
        label="los"
        )
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("Satellite Num")
    plt.ylim(0, limMax)
    # plt.xticks(rotation=15)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.grid(True)

    plt.xlabel("Epoch")
    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

if __name__ == '__main__':
    # rnx = "./data/ml-data/20230511/X6833B-xgboost-1001.rnx"
    # fig = "./data/ml-data/X6833B-satnum.png"
    # data = readrnx(rnx)
    # ref = StatisticResult(data)
    # DrawFigures(ref,fig)

    rnx = "./data/202401/p40-kmeans-0120.rnx"
    fig = "./data/202401/SatNUM/P40-satnum.png"
    data = readrnx(rnx)
    ref = StatisticResult(data)
    DrawFigures(ref,fig)

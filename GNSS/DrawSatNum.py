'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-10-29 21:36:51
LastEditTime: 2025-01-25 19:59:15
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
from com.readrnx import StatisticSatNumResult
from com.mytime import gpsws2ymdhms
from datetime import datetime

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 9  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 9  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 9


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

def DrawSatNumFigure(_satnum1, _satnum2, _satnum3, _figname):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 绘制散点图或折线图
    input      : 类cStat的对象:_stat
    """
    # 时间格式：%Y-%m-%d %H:%M:%S，x轴显示：%H:%M:%S
    ymdhms = np.zeros((len(_satnum1.gpsw), 6), dtype=float)
    Time_hms = []
    for ieph in range(0, len(_satnum1.gpsw)):
        (
            ymdhms[ieph][0],
            ymdhms[ieph][1],
            ymdhms[ieph][2],
            ymdhms[ieph][3],
            ymdhms[ieph][4],
            ymdhms[ieph][5],
        ) = gpsws2ymdhms(
            int(_satnum1.gpsw[ieph]),
            (_satnum1.gpsw[ieph] - int(_satnum1.gpsw[ieph])) * 86400 * 7,
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

    limMax = 60
    mScale = 0.7
    lWidth = 0.8
    # locator = mdates.SecondLocator(30)
    locator = mdates.SecondLocator(bysecond=[0, 30])

    ## draw position
    # plt.figure(dpi=300, figsize=(14.4/Inch, 10/Inch))
    plt.figure(dpi=300,figsize=(4.0,3.5*1.0))
    myFmt = mdates.DateFormatter("%M:%S")
    
    plt.subplot(3, 1, 1)
    plt.plot(
        Time_hms[0 : len(_satnum1.dz)],
        _satnum1.dz,
        color='gray',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="Sat"
        )
    plt.plot(
        Time_hms[0 : len(_satnum1.dy)],
        _satnum1.dy,
        color='red',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="LOS"
        )
    plt.legend(loc="lower right",)
    plt.ylabel("Alloy")
    plt.ylim(0, 33)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距
    
    plt.subplot(3, 1, 2)
    plt.plot(
        Time_hms[0 : len(_satnum2.dz)],
        _satnum2.dz,
        color='gray',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="Sat"
        )
    plt.plot(
        Time_hms[0 : len(_satnum2.dy)],
        _satnum2.dy,
        color='Green',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="LOS"
        )
    plt.legend(loc="lower right",)
    plt.ylabel("ublox")
    plt.ylim(0, 33)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距
    
    plt.subplot(3, 1, 3)
    plt.plot(
        Time_hms[0 : len(_satnum3.dz)],
        _satnum3.dz,
        color='gray',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="Sat"
        )
    plt.plot(
        Time_hms[0 : len(_satnum3.dy)],
        _satnum3.dy,
        color='blue',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="LOS"
        )
    plt.legend(loc="lower right",)
    plt.ylabel("P40")
    plt.ylim(0, 33)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

def DrawSatNumFigure1(_satnum1, _satnum2, _satnum3, _figname):
    # 时间格式：%Y-%m-%d %H:%M:%S，x轴显示：%H:%M:%S
    ymdhms = np.zeros((len(_satnum1.gpsw), 6), dtype=float)
    Time_hms = []
    for ieph in range(0, len(_satnum1.gpsw)):
        (
            ymdhms[ieph][0],
            ymdhms[ieph][1],
            ymdhms[ieph][2],
            ymdhms[ieph][3],
            ymdhms[ieph][4],
            ymdhms[ieph][5],
        ) = gpsws2ymdhms(
            int(_satnum1.gpsw[ieph]),
            (_satnum1.gpsw[ieph] - int(_satnum1.gpsw[ieph])) * 86400 * 7,
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

    limMax = 60
    mScale = 0.7
    lWidth = 0.8
    # locator = mdates.SecondLocator(30)
    locator = mdates.SecondLocator(bysecond=[0, 30])

    ## draw position
    # plt.figure(dpi=300, figsize=(14.4/Inch, 10/Inch))
    plt.figure(dpi=300,figsize=(4.8,4*1.0))
    myFmt = mdates.DateFormatter("%M:%S")
    
    plt.plot(
        Time_hms[0 : len(_satnum1.dz)],
        _satnum1.dz,
        color='gray',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="Alloy-Sat"
        )
    plt.plot(
        Time_hms[0 : len(_satnum1.dy)],
        _satnum1.dy,
        color='red',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="Alloy-LOS"
        )
    
    plt.plot(
        Time_hms[0 : len(_satnum2.dz)],
        _satnum2.dz,
        color='black',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="ublox-Sat"
        )
    plt.plot(
        Time_hms[0 : len(_satnum2.dy)],
        _satnum2.dy,
        color='Green',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="ublox-LOS"
        )
    
    plt.plot(
        Time_hms[0 : len(_satnum3.dz)],
        _satnum3.dz,
        color='brown',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="P40-Sat"
        )
    plt.plot(
        Time_hms[0 : len(_satnum3.dy)],
        _satnum3.dy,
        color='blue',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="P40-LOS"
        )
    plt.legend(loc="lower right",ncol=3,)
    plt.ylabel("Sat.Num")
    plt.ylim(0, 33)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

if __name__ == '__main__':
    # rnx = "./data/ml-data/20230511/X6833B-xgboost-1001.rnx"
    # fig = "./data/ml-data/X6833B-satnum.png"
    # data = readrnx(rnx)
    # ref = StatisticResult(data)
    # DrawFigures(ref,fig)

    # rnx = "./data/202401/p40-kmeans-0120.rnx"
    # fig = "./data/202401/SatNUM/P40-satnum.png"
    # data = readrnx(rnx)
    # ref = StatisticSatNumResult(data)
    # DrawFigures(ref,fig)

    rnx1 = "./data/202401/shadowmatching/alloy-3dma-0120.rnx"
    data1 = readrnx(rnx1)

    rnx2 = "./data/202401/shadowmatching/ublox-3dma-0120.rnx"
    data2 = readrnx(rnx2)

    rnx3 = "./data/202401/shadowmatching/p40-3dma-0120.rnx"
    data3 = readrnx(rnx3)

    ref1 = StatisticSatNumResult(data1)
    ref2 = StatisticSatNumResult(data2)
    ref3 = StatisticSatNumResult(data3)
    # DrawSatNumFigure(ref1,ref2,ref3,"./data/202401/shadowmatching/satnum-all.png")
    DrawSatNumFigure1(ref1,ref2,ref3,"./data/202401/shadowmatching/satnum-all.png")

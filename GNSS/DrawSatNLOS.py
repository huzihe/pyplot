'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-10-29 21:36:51
LastEditTime: 2023-12-11 22:39:42
FilePath: /pyplot/GNSS/DrawSatNLOS.py
Descripttion: 
'''
# -*- coding:utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
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


def DrawFiguresSNR(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(8.4*inch, 5*inch))
    plt.plot(
        los["second"],
        los["SNR"],
        color='C9',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["second"],
        nlos["SNR"],
        color='C1',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("C/N0(dB-Hz)")
    plt.ylim(0, 55)
    plt.grid(True)

    plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresElevation(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(8.4*inch, 5*inch))
    plt.plot(
        los["second"],
        los["elevation"],
        color='C9',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["second"],
        nlos["elevation"],
        color='C1',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("elevation(deg.)")
    plt.ylim(0, 90)
    plt.grid(True)

    plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresPsu(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(8.4*inch, 5*inch))
    plt.plot(
        los["second"],
        los["postResp"],
        color='C9',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["second"],
        nlos["postResp"],
        color='C1',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("psuedorange error(m)")
    plt.ylim(-50, 50)
    plt.grid(True)

    plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresDeltSNR(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(8.4*inch, 5*inch))
    plt.plot(
        los["second"],
        los["priorR"],
        color='C9',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["second"],
        nlos["priorR"],
        color='C1',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("∆C/N0(dB-Hz)")
    plt.ylim(0, 30)
    plt.grid(True)

    plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

if __name__ == '__main__':
    res = "./data/ml-data/20230511/X6833B.res1"
    # snrfig = "./data/ml-data/X6833B-nlos-snr1.png"
    # DrawFiguresSNR(res,snrfig)

    # elefig = "./data/ml-data/X6833B-nlos-elevation1.png"
    # DrawFiguresElevation(res,elefig)

    # psufig = "./data/ml-data/X6833B-nlos-psu1.png"
    # DrawFiguresPsu(res,psufig)

    deltsnrfig = "./data/ml-data/X6833B-nlos-deltsnr.png"
    DrawFiguresDeltSNR(res,deltsnrfig)
    
'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-10-29 21:36:51
LastEditTime: 2024-03-17 22:26:05
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
# from com.readrnx import StatisticResult
from com.mytime import gpsws2ymdhms,gpsws2datetime
from datetime import datetime

locator1 = mdates.MinuteLocator(interval=5)
locator2 = mdates.MinuteLocator(interval=5)

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8

def ws2hms2(second):
    return gpsws2datetime(2261,second)

def DrawFiguresSNR(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    # gnssdata = gnssdata[(base.second<356350)&(base.second>354630)]
    gnssdata = gnssdata[base.second>356350]

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

def DrawFiguresSNR2(_file,_figname):
    gnssdata = pd.read_csv(_file)
    gnssdata['hms']=gnssdata['second'].apply(ws2hms2)
    base = gnssdata[["hms","week","second","los","postResp", "priorR", "elevation", "SNR"]]
    
    # gnssdata_static = gnssdata[(base.second<356350)&(base.second>354630)]
    # gnssdata = gnssdata[base.second>356350]
    gnssdata_static = gnssdata[(base.second<115200)]   # 115200  116040
    gnssdata = gnssdata[(base.second>115200)&(base.second<116460)]


    los_s = gnssdata_static[base.los==1]
    nlos_s = gnssdata_static[base.los==0]

    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(12.9*inch, 4.2*inch))
    plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距
    
    plt.subplot(1,2,1)
    plt.plot(
        los_s["hms"],
        los_s["SNR"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos_s["hms"],
        nlos_s["SNR"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    # plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("C/N0(dB-Hz)")
    plt.ylim(0, 55)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator1)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    ax2 = plt.subplot(1,2,2)
    plt.plot(
        los["hms"],
        los["SNR"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["hms"],
        nlos["SNR"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",markerscale=4,handletextpad=0)
    # plt.ylabel("C/N0(dB-Hz)")
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    plt.ylim(0, 55)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator2)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)


    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresElevation(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    gnssdata = gnssdata[(base.second<356350)&(base.second>354630)]
    # gnssdata = gnssdata[base.second>356350]
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

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresElevation2(_file,_figname):
    gnssdata = pd.read_csv(_file)
    gnssdata['hms']=gnssdata['second'].apply(ws2hms2)
    base = gnssdata[["hms","week","second","los","postResp", "priorR", "elevation", "SNR"]]
    
    # gnssdata_static = gnssdata[(base.second<356350)&(base.second>354630)]
    # gnssdata = gnssdata[base.second>356350]
    gnssdata_static = gnssdata[(base.second<115200)]   # 115200  116040
    gnssdata = gnssdata[(base.second>115200)&(base.second<116460)]


    los_s = gnssdata_static[base.los==1]
    nlos_s = gnssdata_static[base.los==0]

    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(12.9*inch, 4.2*inch))
    plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距
    
    plt.subplot(1,2,1)
    plt.plot(
        los_s["hms"],
        los_s["elevation"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos_s["hms"],
        nlos_s["elevation"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    # plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("elevation(deg.)")
    plt.ylim(0, 90)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator1)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    ax2 = plt.subplot(1,2,2)
    plt.plot(
        los["hms"],
        los["elevation"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["hms"],
        nlos["elevation"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",markerscale=4,handletextpad=0)
    # plt.ylabel("C/N0(dB-Hz)")
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    plt.ylim(0, 90)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator2)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)


    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresPsu(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    # gnssdata = gnssdata[(base.second<356350)&(base.second>354630)]
    gnssdata = gnssdata[base.second>356350]

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

def DrawFiguresPsu2(_file,_figname):
    gnssdata = pd.read_csv(_file)
    gnssdata['hms']=gnssdata['second'].apply(ws2hms2)
    base = gnssdata[["hms","week","second","los","postResp", "priorR", "elevation", "SNR"]]
    
    # gnssdata_static = gnssdata[(base.second<356350)&(base.second>354630)]
    # gnssdata = gnssdata[base.second>356350]
    gnssdata_static = gnssdata[(base.second<115200)]   # 115200  116040
    gnssdata = gnssdata[(base.second>115200)&(base.second<116460)]


    los_s = gnssdata_static[base.los==1]
    nlos_s = gnssdata_static[base.los==0]

    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(12.9*inch, 4.2*inch))
    plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距
    
    plt.subplot(1,2,1)
    plt.plot(
        los_s["hms"],
        los_s["postResp"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos_s["hms"],
        nlos_s["postResp"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    # plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("psuedorange error(m)")
    plt.ylim(-50, 50)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator1)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    ax2 = plt.subplot(1,2,2)
    plt.plot(
        los["hms"],
        los["postResp"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["hms"],
        nlos["postResp"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",markerscale=4,handletextpad=0)
    # plt.ylabel("C/N0(dB-Hz)")
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    plt.ylim(-50, 50)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator2)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)


    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    plt.show()

def DrawFiguresDeltSNR(_file,_figname):
    gnssdata = pd.read_csv(_file)
    base = gnssdata[["second","los","postResp", "priorR", "elevation", "SNR"]]
    gnssdata = gnssdata[(base.second<356350)&(base.second>354630)]
    # gnssdata = gnssdata[base.second>356350]
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

def DrawFiguresDeltSNR2(_file,_figname):
    gnssdata = pd.read_csv(_file)
    gnssdata['hms']=gnssdata['second'].apply(ws2hms2)
    base = gnssdata[["hms","week","second","los","postResp", "priorR", "elevation", "SNR","resSNR"]]
    
    gnssdata_static = gnssdata[(base.second<115200)]   # 115200  116040
    gnssdata = gnssdata[(base.second>115200)&(base.second<116460)]

    los_s = gnssdata_static[base.los==1]
    nlos_s = gnssdata_static[base.los==0]

    los = gnssdata[base.los==1]
    nlos = gnssdata[base.los==0]

    limMax = 50
    inch = 1/2.54

    plt.figure(dpi=300, figsize=(12.9*inch, 4.2*inch))
    plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距
    
    plt.subplot(1,2,1)
    plt.plot(
        los_s["hms"],
        los_s["resSNR"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos_s["hms"],
        nlos_s["resSNR"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    # plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("∆C/N0(dB-Hz)")
    plt.ylim(-30, 20)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator1)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    ax2 = plt.subplot(1,2,2)
    plt.plot(
        los["hms"],
        los["resSNR"],
        color='limegreen',
        linestyle='',
        marker='.',
        markersize='1',
        label="los"
        )
    plt.plot(
        nlos["hms"],
        nlos["resSNR"],
        color='deeppink',
        linestyle='',
        marker='.',
        markersize='1',
        label="nlos"
        )
    plt.legend(loc="lower right",markerscale=4,handletextpad=0)
    # plt.ylabel("C/N0(dB-Hz)")
    ax2.yaxis.set_major_formatter(plt.NullFormatter())
    plt.ylim(-30, 20)
    plt.grid(True)
    # plt.xlabel("Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置 x 轴为日期格式
    plt.gca().xaxis.set_major_locator(locator2)  # x 轴间隔
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.savefig(_figname, bbox_inches="tight") #pad_inches=0.2*inch
    # plt.show()


if __name__ == '__main__':
    res = "./data/ml-data/20230511/X6833B.res1"
    # snrfig = "./data/ml-data/X6833B-nlos-snr.png"
    # DrawFiguresSNR(res,snrfig)

    # elefig = "./data/ml-data/X6833B-nlos-elevation-s.png"
    # DrawFiguresElevation(res,elefig)

    # psufig = "./data/ml-data/X6833B-nlos-psu-d.png"
    # DrawFiguresPsu(res,psufig)

    # deltsnrfig = "./data/ml-data/X6833B-nlos-deltsnr-s.png"
    # DrawFiguresDeltSNR(res,deltsnrfig)

    # snrfig = "./data/ml-data/X6833B-nlos-snr3.png"
    # DrawFiguresSNR2(res,snrfig)

    # elefig = "./data/ml-data/X6833B-nlos-elevation3.png"
    # DrawFiguresElevation2(res,elefig)

    # psufig = "./data/ml-data/X6833B-nlos-psu3.png"
    # DrawFiguresPsu2(res,psufig)

    res = "./data/202401/log-spp-ublox.res1"
    psufig = "./data/ml-data/ublox-nlos-deltSNR3.png"
    DrawFiguresDeltSNR2(res,psufig)

    snrfig = "./data/ml-data/ublox-nlos-snr3.png"
    DrawFiguresSNR2(res,snrfig)

    elefig = "./data/ml-data/ublox-nlos-elevation3.png"
    DrawFiguresElevation2(res,elefig)

    psufig = "./data/ml-data/ublox-nlos-psu3.png"
    DrawFiguresPsu2(res,psufig)

    # deltsnrfig = "./data/ml-data/X6833B-nlos-deltsnr-s.png"
    # DrawFiguresDeltSNR(res,deltsnrfig)

    # res = "./data/ml-data/20230511/trimble.res1"
    # snrfig = "./data/ml-data/trimble-nlos-snr1.png"
    # DrawFiguresSNR(res,snrfig)

    # elefig = "./data/ml-data/trimble-nlos-elevation1.png"
    # DrawFiguresElevation(res,elefig)

    # psufig = "./data/ml-data/trimble-nlos-psu1.png"
    # DrawFiguresPsu(res,psufig)

    # deltsnrfig = "./data/ml-data/trimble-nlos-deltsnr1.png"
    # DrawFiguresDeltSNR(res,deltsnrfig)
    
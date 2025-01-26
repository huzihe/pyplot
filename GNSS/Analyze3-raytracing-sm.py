#!/usr/bin/python3
"""
@author    : shengyixu Created on 2022.8.20
Purpose    : 绘制PosGO软件"动态"残差序列, 参考值来自IE
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from com.mytime import gpsws2ymdhms
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import radians, sin,fabs,cos,asin,sqrt
from com.com import std, rms, mean, maxabs, get_distance_hav, get_distance_hav, get_deltB, get_deltL
from com.readfile import ReadMyResult, ReadIERefResult, ReadGINSResult, Read3DMAResult
from com.readrnx import readrnx
from com.readrnx import StatisticSatNumResult

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
# plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
# plt.rcParams['axes.labelsize'] = 7  # 坐标轴标签字体大小
# plt.rcParams['xtick.labelsize'] = 7  # x轴刻度字体大小
# plt.rcParams['ytick.labelsize'] = 7  # y轴刻度字体大小
# plt.rcParams['legend.fontsize'] = 6
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 9  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 9  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 9

Inch = 2.54

class cStat(object):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 残差序列以及数值特征
    """

    def __init__(self):
        self.gpsw = []
        self.dx, self.dy, self.dz = [], [], []
        self.rms = []
        self.std = []
        self.max = []
        self.mean = []
        self.fixratio = 0


RE = 6378137


def CalDifference(_cal, _ref):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 获取PosGO计算值与IE参考值的残差序列
    input      : PosGO计算文件:_cal
                    IE参考文件:_ref
    """
    result = {}
    for time, cal in _cal.items():
        if time in _ref.keys():
            result.update(
                {
                    time: {
                        "b": get_deltB(cal["b"],cal["l"],_ref[time]["b"],_ref[time]["l"]),
                        "l": get_deltL(cal["b"],cal["l"],_ref[time]["b"],_ref[time]["l"]),
                        # "b": get_distance_hav(cal["b"],cal["l"],_ref[time]["b"],cal["l"]),
                        # "l": get_distance_hav(_ref[time]["b"],cal["l"],_ref[time]["b"],_ref[time]["l"]),
                        "h": cal["h"] - _ref[time]["h"],
                        "stat": cal["stat"],
                    }
                }
            )
            b = get_distance_hav(cal["b"],cal["l"],_ref[time]["b"],cal["l"])
            l = get_distance_hav(_ref[time]["b"],cal["l"],_ref[time]["b"],_ref[time]["l"])
            dist = get_distance_hav(cal["b"],cal["l"],_ref[time]["b"],_ref[time]["l"])
    return result


def StatisticResult(_det):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 统计残差序列的数值特征
    input      : 残差序列:_det
    """
    all, fix = 0, 0
    border = 100
    _stat = cStat()
    for time, det in _det.items():
        all += 1
        _stat.gpsw.append(time)
        if abs(det["b"]) < border:
            _stat.dx.append(det["b"])
        if abs(det["l"]) < border:
            _stat.dy.append(det["l"])
        if abs(det["h"]) < border:
            _stat.dz.append(det["h"])
        if det["stat"] == 1 or det["stat"] == 3:
            fix += 1

    _stat.std = [std(_stat.dx) * 100, std(_stat.dy) * 100, std(_stat.dz) * 100]
    _stat.rms = [rms(_stat.dx) * 100, rms(_stat.dy) * 100, rms(_stat.dz) * 100]
    _stat.mean = [mean(_stat.dx) * 100, mean(_stat.dy) * 100, mean(_stat.dz) * 100]

    _stat.dx.clear()
    _stat.dy.clear()
    _stat.dz.clear()

    # 重新保存残差序列
    for time, det in _det.items():
        _stat.dx.append(det["b"])
        _stat.dy.append(det["l"])
        _stat.dz.append(det["h"])

    _stat.max = [maxabs(_stat.dx) * 100, maxabs(_stat.dy) * 100, maxabs(_stat.dz) * 100]

    print("Position Error (BLH cm):")
    print("RMS   %9.3f    %9.3f    %9.3f" % (_stat.rms[0], _stat.rms[1], _stat.rms[2]))
    print("STD   %9.3f    %9.3f    %9.3f" % (_stat.std[0], _stat.std[1], _stat.std[2]))
    print("MAX   %9.3f    %9.3f    %9.3f" % (_stat.max[0], _stat.max[1], _stat.max[2]))
    print(
        "MEAN  %9.3f    %9.3f    %9.3f" % (_stat.mean[0], _stat.mean[1], _stat.mean[2])
    )

    print(
        "All epoch:%7d, Fix epoch:%7d, Percentage:%7.3f" % (all, fix, fix / all * 100)
    )

    _stat.fixratio = fix / all * 100

    return _stat


def ExportDifference(_file, _det):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 将残差序列输出到文件中
    input      : 残差序列:_det
    output     : 残差文件:_file
    """
    with open(_file, "w") as file:
        for time, each in _det.items():
            file.write(
                "%10.6f, %9.3f, %9.3f, %9.3f, %2d \n"
                % (time, each["b"], each["l"], each["h"], each["stat"])
            )


def DrawFigure(_stat, _stat2, _stat3, _figname):
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

    limMax = 60
    ## draw position
    plt.figure(dpi=300, figsize=(14.9/Inch, 7/Inch))
    myFmt = mdates.DateFormatter("%H:%M:%S")
    plt.subplot(3, 1, 1)
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat3.dy,
        color='grey',linestyle='', marker='.',markersize='2',
        label=str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" spp" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dy)],
        _stat2.dy,
        color='deeppink',linestyle='', marker='.',markersize='2',
        label=str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + "     spp-xgb",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dy)],
        _stat.dy,
        color='limegreen',linestyle='', marker='.',markersize='1.5',
        label=str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + "     spp-kmeans",
    )
    #   plt.plot(Time_hms[0:len(_stat.dy)], _stat.dy, 'red', label = 'rms_L: ' + str(round(_stat.rms[1], 3)) + 'cm')
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.legend(loc="upper left",bbox_to_anchor=(-0.05,1.8),markerscale=2, handletextpad=0,frameon=False)
    plt.ylabel("East(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 5:
        plt.ylim(-limMax, limMax)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    # plt.plot(Time_hms[0 : len(_stat.dx)],_stat.dx,color='C1',linestyle='', marker='.',markersize='2',label="satellite")

    plt.subplot(3, 1, 2)
    plt.plot(
        Time_hms[0 : len(_stat3.dx)],
        _stat3.dx,
        color='grey',linestyle='', marker='.',markersize='2',
        # label=str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" spp" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat2.dx,
        color='deeppink',linestyle='', marker='.',markersize='2',
        # label=str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " spp-xgb",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dx,
        color='limegreen',linestyle='', marker='.',markersize='1.5',
        # label=str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " spp-kmeans",
    )
    #   plt.plot(Time_hms[0:len(_stat.dx)], _stat.dx, 'blue', label = 'rms_B: ' + str(round(_stat.rms[0], 3)) + 'cm')
    plt.ylabel("North(m)")
    if max(_stat.max[0:2]) / 100 > 5:
        plt.ylim(-limMax, limMax)
    else:
        plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    # plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.subplot(3, 1, 3)
    plt.plot(
        Time_hms[0 : len(_stat2.dz)],
        _stat3.dz,
        color='grey',linestyle='', marker='.',markersize='2',
        # label="spp: " + str(round(_stat3.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dz)],
        _stat2.dz,
        color='deeppink',linestyle='', marker='.',markersize='2',
        # label="spp_xgb: " + str(round(_stat2.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dz)],
        _stat.dz,
        color='limegreen',linestyle='', marker='.',markersize='1.5',
        # label="spp_kmeans: " + str(round(_stat.rms[2]/100, 3)) + "m",
    )
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.ylabel("Up(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 5:
        plt.ylim(-limMax, limMax)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.grid(True)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

def DrawFigureLine(_stat, _stat2, _stat3, _figname):
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

    limMax = 60
    ## draw position
    plt.figure(dpi=300, figsize=(8.4/Inch, 5/Inch))
    myFmt = mdates.DateFormatter("%M:%S")
    plt.subplot(3, 1, 1)
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat3.dy,
        color='grey',linestyle='-', linewidth='0.8', #marker='.',markersize='2',
        label=str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" SPP" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dy)],
        _stat2.dy,
        color='deeppink',linestyle='-', linewidth='0.8', #marker='.',markersize='2',
        label=str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " SPP-SM",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dy)],
        _stat.dy,
        color='limegreen',linestyle='-', linewidth='0.8', #marker='.',markersize='1.5',
        label=str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " SPP-RT",
    )
    #   plt.plot(Time_hms[0:len(_stat.dy)], _stat.dy, 'red', label = 'rms_L: ' + str(round(_stat.rms[1], 3)) + 'cm')
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.legend(loc="upper left",bbox_to_anchor=(0.005,1.85),markerscale=1, handletextpad=0.5,frameon=False)
    plt.ylabel("East(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    # plt.plot(Time_hms[0 : len(_stat.dx)],_stat.dx,color='C1',linestyle='', marker='.',markersize='2',label="satellite")

    plt.subplot(3, 1, 2)
    plt.plot(
        Time_hms[0 : len(_stat3.dx)],
        _stat3.dx,
        color='grey',linestyle='-', linewidth='0.8', #marker='.',markersize='2',
        # label=str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" spp" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat2.dx,
        color='deeppink',linestyle='-', linewidth='0.8', #marker='.',markersize='2',
        # label=str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " spp-xgb",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dx,
        color='limegreen',linestyle='-', linewidth='0.8', #marker='.',markersize='1.5',
        # label=str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " spp-kmeans",
    )
    plt.ylabel("North(m)")
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    else:
        plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    # plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.subplot(3, 1, 3)
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat3.dz,
        color='grey',linestyle='-', linewidth='0.8', #marker='.',markersize='2',
        # label="spp: " + str(round(_stat3.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dz)],
        _stat2.dz,
        color='deeppink',linestyle='-', linewidth='0.8', #marker='.',markersize='2',
        # label="spp_xgb: " + str(round(_stat2.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dz)],
        _stat.dz,
        color='limegreen',linestyle='-', linewidth='0.8', #marker='.',markersize='1.5',
        # label="spp_kmeans: " + str(round(_stat.rms[2]/100, 3)) + "m",
    )
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.ylabel("Up(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.grid(True)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

def DrawFigureLineWithSat(_stat, _stat2, _stat3, _satnum1, _figname):
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

    limMax = 60
    mScale = 0.7
    lWidth = 0.8
    # locator = mdates.SecondLocator(30)
    locator = mdates.SecondLocator(bysecond=[0, 30])

    ## draw position
    # plt.figure(dpi=300, figsize=(14.4/Inch, 10/Inch))
    plt.figure(dpi=300,figsize=(4.0,4*1.0))
    myFmt = mdates.DateFormatter("%M:%S")
    plt.subplot(4, 1, 1)
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat3.dy,
        color='grey', linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        label=str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" SPP" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dy)],
        _stat2.dy,
        color='deeppink', linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        label=str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " SPP-SM",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dy)],
        _stat.dy,
        color='limegreen', linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        label=str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " SPP-RT",
    )
    #   plt.plot(Time_hms[0:len(_stat.dy)], _stat.dy, 'red', label = 'rms_L: ' + str(round(_stat.rms[1], 3)) + 'cm')
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.legend(loc="upper left",bbox_to_anchor=(0.005,1.8),markerscale=1, handletextpad=0.5,frameon=False)
    plt.ylabel("dE(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    # plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    # plt.plot(Time_hms[0 : len(_stat.dx)],_stat.dx,color='C1',linestyle='', marker='.',markersize='2',label="satellite")

    plt.subplot(4, 1, 2)
    plt.plot(
        Time_hms[0 : len(_stat3.dx)],
        _stat3.dx,
        color='grey',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label=str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" spp" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat2.dx,
        color='deeppink',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label=str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " spp-xgb",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dx,
        color='limegreen',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label=str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " spp-kmeans",
    )
    plt.ylabel("dN(m)")
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    else:
        plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    # plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.subplot(4, 1, 3)
    plt.plot(
        Time_hms[0 : len(_stat3.dz)],
        _stat3.dz,
        color='grey',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label="spp: " + str(round(_stat3.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dz)],
        _stat2.dz,
        color='deeppink',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label="spp_xgb: " + str(round(_stat2.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dz)],
        _stat.dz,
        color='limegreen',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label="spp_kmeans: " + str(round(_stat.rms[2]/100, 3)) + "m",
    )
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.ylabel("dU(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-95, 95)
    plt.yticks(np.arange(-80, 80+0.1, 80))
    # plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.subplot(4, 1, 4)
    plt.plot(
        Time_hms[0 : len(_satnum1.dz)],
        _satnum1.dz,
        color='red',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="All"
        )
    plt.plot(
        Time_hms[0 : len(_satnum1.dy)],
        _satnum1.dy,
        color='blue',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="LOS"
        )
    plt.legend(loc="lower right",)
    plt.ylabel("Sat. Num")
    plt.ylim(0, 30)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

def DrawFigureLineWithSat1(_stat, _stat2, _stat3, _satnum1, _figname):
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

    limMax = 60
    mScale = 0.7
    lWidth = 0.8
    # locator = mdates.SecondLocator(30)
    locator = mdates.SecondLocator(bysecond=[0, 30])

    ## draw position
    # plt.figure(dpi=300, figsize=(14.4/Inch, 10/Inch))
    plt.figure(dpi=300,figsize=(4.0,4*1.0))
    myFmt = mdates.DateFormatter("%M:%S")
    plt.subplot(4, 1, 1)
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat3.dy,
        color='grey', linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        label = "SPP" ,
        # label=str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" SPP" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dy)],
        _stat2.dy,
        color='deeppink', linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        label = "NLOS-W" ,
        # label=str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " SPP-SM",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dy)],
        _stat.dy,
        color='limegreen', linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        label = "SPP-RT" ,
        # label=str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " SPP-RT",
    )
    #   plt.plot(Time_hms[0:len(_stat.dy)], _stat.dy, 'red', label = 'rms_L: ' + str(round(_stat.rms[1], 3)) + 'cm')
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.legend(loc="upper left",bbox_to_anchor=(0,1.35),ncol=3,markerscale=1.5, handletextpad=0,columnspacing=0.5, frameon=False)
    # plt.legend(loc="upper left",bbox_to_anchor=(0.005,1.8),markerscale=1, handletextpad=0.5,frameon=False)
    plt.ylabel("dE(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    # plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    # plt.plot(Time_hms[0 : len(_stat.dx)],_stat.dx,color='C1',linestyle='', marker='.',markersize='2',label="satellite")

    plt.subplot(4, 1, 2)
    plt.plot(
        Time_hms[0 : len(_stat3.dx)],
        _stat3.dx,
        color='grey',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label=str(round(_stat3.rms[0]/100, 2)) + "/" + str(round(_stat3.rms[1]/100, 2)) + "/" + str(round(_stat3.rms[2]/100, 2)) +" spp" ,
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat2.dx,
        color='deeppink',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label=str(round(_stat2.rms[0]/100, 2)) + "/" + str(round(_stat2.rms[1]/100, 2)) + "/" + str(round(_stat2.rms[2]/100, 2)) + " spp-xgb",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dx,
        color='limegreen',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label=str(round(_stat.rms[0]/100, 2)) + "/" + str(round(_stat.rms[1]/100, 2)) + "/" + str(round(_stat.rms[2]/100, 2)) + " spp-kmeans",
    )
    plt.ylabel("dN(m)")
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-limMax, limMax)
    else:
        plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    # plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.subplot(4, 1, 3)
    plt.plot(
        Time_hms[0 : len(_stat3.dz)],
        _stat3.dz,
        color='grey',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label="spp: " + str(round(_stat3.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat2.dz)],
        _stat2.dz,
        color='deeppink',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label="spp_xgb: " + str(round(_stat2.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dz)],
        _stat.dz,
        color='limegreen',linestyle='-', linewidth= lWidth, marker='o',markersize= mScale,
        # label="spp_kmeans: " + str(round(_stat.rms[2]/100, 3)) + "m",
    )
    # plt.legend(loc="lower right",ncol=3,handletextpad=0)
    plt.ylabel("dU(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 2:
        plt.ylim(-95, 95)
    plt.yticks(np.arange(-80, 80+0.1, 80))
    # plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.subplot(4, 1, 4)
    plt.plot(
        Time_hms[0 : len(_satnum1.dz)],
        _satnum1.dz,
        color='red',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="All"
        )
    plt.plot(
        Time_hms[0 : len(_satnum1.dy)],
        _satnum1.dy,
        color='blue',
        linestyle='-',linewidth= lWidth, marker='o',markersize= mScale,
        label="LOS"
        )
    plt.legend(loc="lower right",)
    plt.ylabel("Sat. Num")
    plt.ylim(0, 30)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.subplots_adjust(wspace =0, hspace =0.05)#调整子图间距

    # plt.xlabel("Time")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("#usage  :  Analyze.py result1_file result2_file result3_file ref_file")
        print("#example:  Analyze.py re1.txt re2.txt re3.txt ref.txt")
        sys.exit(0)

    if len(sys.argv) == 5:
        calFile1 = sys.argv[1]
        calFile2 = sys.argv[2]
        calFile3 = sys.argv[3]
        refFile = sys.argv[4]
    filename = calFile1.split(".")[0]
    deltname ="-sw-sat-0125"  # 文件标记
    detFile = "det-" + filename + deltname + ".txt"
    detFile2 = "det-" + calFile2.split(".")[0] + deltname +".txt"
    detFile3 = "det-" + calFile3.split(".")[0] + deltname +".txt"
    figName = "fig-" + filename + deltname +".png"

    calValue = ReadMyResult(calFile1)
    calValue2 = ReadMyResult(calFile2)
    calValue3 = ReadMyResult(calFile3)

    refValue = ReadGINSResult(refFile)

    detValue = CalDifference(calValue, refValue)
    Stat = StatisticResult(detValue)

    detValue2 = CalDifference(calValue2, refValue)
    stat2 = StatisticResult(detValue2)

    detValue3 = CalDifference(calValue3, refValue)
    stat3 = StatisticResult(detValue3)

    ExportDifference(detFile, detValue)
    ExportDifference(detFile2, detValue2)
    ExportDifference(detFile3, detValue3)
    # DrawFigureLine(Stat, stat2, stat3, figName)
    
    # 带卫星数的统计
    nlosrnx = "alloy-3dma-0120.rnx"
    # nlosrnx = "ublox-3dma-0120.rnx"
    # nlosrnx = "p40-3dma-0120.rnx"
    nlosd = readrnx(nlosrnx)
    nlosSatnum = StatisticSatNumResult(nlosd)
    # nlosSatnum.dx.pop()
    # nlosSatnum.dy.pop()
    # nlosSatnum.dz.pop()


    DrawFigureLineWithSat(Stat, stat2, stat3, nlosSatnum, figName)
    # DrawFigureLineWithSat1(Stat, stat2, stat3, nlosSatnum, figName)

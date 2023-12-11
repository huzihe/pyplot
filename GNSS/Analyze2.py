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

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8

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
    _stat = cStat()
    for time, det in _det.items():
        all += 1
        _stat.gpsw.append(time)

        if abs(det["b"]) < 200:
            _stat.dx.append(det["b"])
        if abs(det["l"]) < 200:
            _stat.dy.append(det["l"])
        if abs(det["h"]) < 200:
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


def DrawFigure(_stat, _stat2, _figname):
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

    limMax = 120
    ## draw position
    plt.figure(dpi=600, figsize=(12.9/Inch, 8/Inch))
    myFmt = mdates.DateFormatter("%H:%M:%S")
    plt.subplot(3, 1, 1)
    # plt.plot(Time_hms[0 : len(_stat.dx)],_stat.dx,color='C1',linestyle='', marker='.',markersize='2',label="satellite")
    plt.plot(
        Time_hms[0 : len(_stat2.dx)],
        _stat2.dx,
        color='C1',linestyle='', marker='.',markersize='2',
        label="lsq: " + str(round(_stat2.rms[0]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dx)],
        _stat.dx,
        color='C9',linestyle='', marker='.',markersize='1.5',
        label="lsq_xgb: " + str(round(_stat.rms[0]/100, 3)) + "m",
    )
    #   plt.plot(Time_hms[0:len(_stat.dx)], _stat.dx, 'blue', label = 'rms_B: ' + str(round(_stat.rms[0], 3)) + 'cm')
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("Latitude(m)")
    if max(_stat.max[0:2]) / 100 > 5:
        plt.ylim(-limMax, limMax)
    else:
        plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    # plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(
        Time_hms[0 : len(_stat2.dy)],
        _stat2.dy,
        color='C1',linestyle='', marker='.',markersize='2',
        label="lsq: " + str(round(_stat2.rms[1]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dy)],
        _stat.dy,
        color='C9',linestyle='', marker='.',markersize='1.5',
        label="lsq_xgb: " + str(round(_stat.rms[1]/100, 3)) + "m",
    )
    #   plt.plot(Time_hms[0:len(_stat.dy)], _stat.dy, 'red', label = 'rms_L: ' + str(round(_stat.rms[1], 3)) + 'cm')
    plt.legend(loc="upper right",handletextpad=0)
    plt.ylabel("Longitude(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 5:
        plt.ylim(-limMax, limMax)
    plt.gca().xaxis.set_ticklabels([])
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(
        Time_hms[0 : len(_stat2.dz)],
        _stat2.dz,
        color='C1',linestyle='', marker='.',markersize='2',
        label="lsq: " + str(round(_stat2.rms[2]/100, 3)) + "m",
    )
    plt.plot(
        Time_hms[0 : len(_stat.dz)],
        _stat.dz,
        color='C9',linestyle='', marker='.',markersize='1.5',
        label="lsq_xgb: " + str(round(_stat.rms[2]/100, 3)) + "m",
    )

    #   plt.plot(Time_hms[0:len(_stat.dz)], _stat.dz, 'green', label = 'rms_H: ' + str(round(_stat.rms[2], 3)) + 'cm')
    plt.legend(loc="lower right",handletextpad=0)
    plt.ylabel("Height(m)")
    plt.ylim(-max(_stat.max[0:1]) / 100, max(_stat.max[0:1]) / 100)
    if max(_stat.max[0:2]) / 100 > 5:
        plt.ylim(-limMax, limMax)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.grid(True)

    plt.xlabel("Epoch")
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("#usage  :  Analyze.py result1_file result2_file ref_file")
        print("#example:  Analyze.py re1.txt re2.txt ref.txt")
        sys.exit(0)

    if len(sys.argv) == 4:
        calFile1 = sys.argv[1]
        calFile2 = sys.argv[2]
        refFile = sys.argv[3]
    filename = calFile1.split(".")[0]
    detFile = "det-" + filename + ".txt"
    figName = "fig-" + filename + "-alldynamic.png"

    # calValue = Read3DMAResult(calFile1)
    # calValue2 = Read3DMAResult(calFile2)
    calValue = ReadMyResult(calFile1)
    calValue2 = ReadMyResult(calFile2)
    # calValue = ReadMyResult(calFile)
    # refValue = ReadIERefResult(refFile)
    refValue = ReadGINSResult(refFile)

    detValue = CalDifference(calValue, refValue)
    Stat = StatisticResult(detValue)

    detValue2 = CalDifference(calValue2, refValue)
    stat2 = StatisticResult(detValue2)

    ExportDifference(detFile, detValue)
    DrawFigure(Stat, stat2, figName)

'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-08-13 21:05:21
LastEditTime: 2023-11-04 16:00:25
FilePath: /pyplot/com/readrnx.py
Descripttion: 
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from mytime import ymdhms2gpsws
# from com import std, rms, mean, maxabs

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

def readrnx(rnx):
    result = {}
    time = 0
    with open(rnx, "r") as file:
        content = file.readlines()
        headerflag = False
        for eachLine in content:
            if eachLine.startswith("%"):  # ignore comment line
                continue
            ender = "END OF HEADER" in eachLine
            if ender:
                headerflag = True
                continue
            elif headerflag:
                if eachLine.startswith(">"):
                    eachData = eachLine.split()
                    ws = ymdhms2gpsws(
                        int(eachData[1]),
                        int(eachData[2]),
                        int(eachData[3]),
                        int(eachData[4]),
                        int(eachData[5]),
                        int(float(eachData[6])),
                    )
                    time = int(ws[0]) + float(ws[1]) / 86400.0 / 7.0
                    if 361008 >= float(ws[1])>= 360717:
                    # if time >0:
                        result.update(
                            {
                                (time): {
                                    "nsat":nsat,
                                    "los": los
                                    }
                                })
                    los = 0
                    nsat = int(eachData[8])
                    continue
                else:
                    # eachData = eachLine.split()
                    for itr in range(6):
                        if len(eachLine) > 48 * itr + 18 + 16:
                            substr = eachLine[48 * itr + 18]
                            if substr == '1':
                                los = los + 1
                        else:
                            break
                    continue
            else:
                continue
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

        # _stat.dx.append(det)

        if abs(det["nsat"]) < 100:
            _stat.dx.append(det["nsat"])
        if abs(det["los"]) < 100:
            _stat.dy.append(det["los"])

    # _stat.std = [std(_stat.dx) * 100, std(_stat.dy) * 100, std(_stat.dz) * 100]
    # _stat.rms = [rms(_stat.dx) * 100, rms(_stat.dy) * 100, rms(_stat.dz) * 100]
    # _stat.mean = [mean(_stat.dx) * 100, mean(_stat.dy) * 100, mean(_stat.dz) * 100]

    _stat.dx.clear()
    _stat.dy.clear()

    # 重新保存残差序列
    for time, det in _det.items():
        _stat.dx.append(det["nsat"])
        _stat.dy.append(det["los"])

    return _stat
    

if __name__ == '__main__':
    rnx = "./data/ml-data/20230511/Ck6n-xgboost-1005.rnx"
    stat = readrnx(rnx)
    te=1
'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-08-13 21:05:21
LastEditTime: 2025-01-25 17:39:39
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
                    # if 533622 >= float(ws[1]) >= 529522:   # 20240120 动态
                    # if 361008 >= float(ws[1])>= 360717:
                    # if 529895 >= float(ws[1]) >= 529801: # 20240120 动态  劝业场
                    # if 530362 >= float(ws[1]) >= 530266: # 20240120 动态  中南路
                    # if 532239 >= float(ws[1]) >= 532212:   # 20240120 动态  凌波门
                    # if 532239 >= float(ws[1]) >= 532158:   # 20240120 动态  东湖南路
                    # if 534035 >= float(ws[1]) >= 533671:     # 20140120 动态 星湖大楼四周
                    if 536020 >= float(ws[1]) >= 534035:     # 20140120 静态 星湖大楼东侧门口
                    # if time >0:
                        result.update(
                            {
                                (time): {
                                    "nsat":nsat,
                                    "los": los, 
                                    "nlos": nlos,
                                    # "los": realsat-los-2,
                                    "realSat":realsat
                                    }
                                })
                    los = 0
                    nlos = 0
                    realsat = 0
                    nsat = int(eachData[8])
                    continue
                else:
                    # eachData = eachLine.split()
                    substr = eachLine[0]
                    if substr == 'G' or substr == 'E' or substr == 'R' or substr == 'C':
                        realsat = realsat + 1
                        for itr in range(6):
                            if len(eachLine) > 48 * itr + 18 + 16:
                                substr = eachLine[48 * itr + 18]
                                if substr == '1':
                                    los = los + 1
                                    break
                                if substr == "0":
                                    nlos = nlos + 1
                                    break
                            else:
                                break
                        continue
            else:
                continue
    return result

def StatisticSatNumResult(_det):
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
            _stat.max.append(det["nlos"])
        if abs(det["los"]) < 100:
            _stat.dy.append(det["los"])
        if abs(det["realSat"]) < 100:
            _stat.dz.append(det["realSat"])
            

    # _stat.std = [std(_stat.dx) * 100, std(_stat.dy) * 100, std(_stat.dz) * 100]
    # _stat.rms = [rms(_stat.dx) * 100, rms(_stat.dy) * 100, rms(_stat.dz) * 100]
    # _stat.mean = [mean(_stat.dx) * 100, mean(_stat.dy) * 100, mean(_stat.dz) * 100]

    _stat.dx.clear()
    _stat.dy.clear()
    _stat.dz.clear()
    _stat.max.clear()
    _stat.mean.clear()

    # 重新保存残差序列
    for time, det in _det.items():
        _stat.dx.append(det["nsat"])
        _stat.dy.append(det["los"])
        _stat.dz.append(det["realSat"])
        _stat.max.append(det["nlos"])
        _stat.mean.append(det["realSat"]- det["nlos"]+3)

    return _stat
    

if __name__ == '__main__':
    rnx = "./data/ml-data/20230511/Ck6n-xgboost-1005.rnx"
    stat = readrnx(rnx)
    te=1
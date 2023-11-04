"""
Author: hzh huzihe@whu.edu.cn
Date: 2022-08-25 14:12:54
LastEditTime: 2023-06-03 20:16:43
FilePath: /POSGO_V1.0/script/readfile.py
Descripttion: 
"""
"""
@author    : shengyixu Created on 2022.8.20
Purpose    : 读取各软件输出的结果文件
"""


def ReadGINSResult(_file):
    result = {}
    with open(_file, "r") as file:
        content = file.readlines()
        for eachLine in content:
            if eachLine.startswith("%"):  # ignore comment line
                continue
            eachData = eachLine.split()
            # GPSweek(int) + GPSsecond => week(float)
            time = int(eachData[0]) + float(eachData[1]) / 86400.0 / 7.0

            #            if (float(eachData[8]) > 7):
            #                continue
            result.update(
                {
                    (time): {
                        "b": float(eachData[2]),
                        "l": float(eachData[3]),
                        "h": float(eachData[4]),
                    }
                }
            )
    return result


def Read3DMAResult(_file):
    """
    @author    : huzihe Create on 2023.6.3
    Purpose    : read the gnss-3dma results
    """
    result = {}
    with open(_file, "r") as file:
        content = file.readlines()
        for eacheline in content:
            if eacheline.startswith("%"):
                continue
            eachData = eacheline.split()
            time = int(eachData[0]) + float(eachData[1]) / 86400.0 / 7.0

            result.update(
                {
                    time: {
                        "b": float(eachData[2]),
                        "l": float(eachData[3]),
                        "h": float(eachData[4]),
                        # "num": int(eachData[7]),
                        "stat": int(2),
                    }
                }
            )

    return result


def ReadMyResult(_file):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 读取PosGO软件输出的结果文件
    """
    result = {}
    with open(_file, "r") as file:
        content = file.readlines()
        for eachLine in content:
            if eachLine.startswith("%"):
                continue
            eachData = eachLine.split(",")
            # GPSweek(int) + GPSsecond => week(float)
            time = int(eachData[0]) + int(float(eachData[1])) / 86400.0 / 7.0
            # if float(eachData[1]) >= 356340:    #2023 5 11 2 59 0
            # if float(eachData[1]) >= 357120:    #2023 5 11 3 12 0
            # if 361008 >= float(eachData[1])>= 360717:
            result.update(
                {
                    time: {
                        "b": float(eachData[2]),
                        "l": float(eachData[3]),
                        "h": float(eachData[4]),
                        "num": float(eachData[9]),
                        #  'ratio':float(eachData[11]),
                        "stat": float(eachData[8]),
                    }
                }
            )
    return result


def ReadIERefResult(_file):
    """
    @author    : shengyixu Created on 2022.8.20
    Purpose    : 读取IE软件输出的结果文件
    """
    result = {}
    with open(_file, "r") as file:
        content = file.readlines()
        for eachLine in content:
            if eachLine.startswith("%"):
                continue
            eachData = eachLine.split(",")
            # GPSweek(int) + GPSsecond => week(float)
            time = int(eachData[0]) + float(eachData[1]) / 86400.0 / 7.0

            result.update(
                {
                    time: {
                        "b": float(eachData[2]),
                        "l": float(eachData[3]),
                        "h": float(eachData[4]),
                        "vb": float(eachData[6]),
                        "vl": float(eachData[5]),
                        "vh": float(eachData[7]),
                        "stat": float(eachData[13]),
                    }
                }
            )  # note stat shoule be 14
    return result

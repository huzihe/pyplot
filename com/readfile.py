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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from mytime import ymdhms2gpsws


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

def Read3DMAResult2(_file):
    """
    @author    : huzihe Create on 2024.11.24
    Purpose    : read the gnss-3dma results
    """
    result = {}
    with open(_file, "r") as file:
        content = file.readlines()
        for eacheline in content:
            if eacheline.startswith("%"):
                continue
            eachData = eacheline.split(",")
            time = int(eachData[0]) + float(eachData[1]) / 86400.0 / 7.0
            # if 529458 >= float(eachData[1]) >= 529020:   # 20240120 动态 星湖大楼四周绕圈 "02:55:00"-"03:04:00"
            # if 534034 >= float(eachData[1]):   # 20240120 动态 星湖大楼四周绕圈 回程
            # if 536047 >= float(eachData[1]) >= 534035:   # 静态 回程
            # if 533694 >= float(eachData[1]) >= 533681:   # 动态 西区宿舍南北向
            # if 533734 >= float(eachData[1]) >= 533721:   # 动态 国软南北向
            if 536020 >= float(eachData[1]) >= 533670:     # 20140120 动态 星湖大楼四周+东侧门口
                result.update(
                    {
                        time: {
                            "b": float(eachData[5]),
                            "l": float(eachData[6]),
                            "h": float(eachData[7]),
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
            # if 354600 <= float(eachData[1]) <= 356340:    #2023-5-11 2:30:00——2:59:00
            # if 356340 <= float(eachData[1]) <= 356980:    #2023-5-11 2:59:00——3:09:00
            # if float(eachData[1]) >= 357120:    #2023 5 11 3 12 0
            # if float(eachData[1]) >= 357580:    #妇幼门口开始
            # if 358670 >= float(eachData[1])>= 358520:
            # if 358607 >= float(eachData[1])>= 358606:
            # if float(eachData[1])<= 356340 or float(eachData[1])>= 357120:
            # if 533622 >= float(eachData[1]) >= 529522:   # 20240120 动态
            # if 534034 >= float(eachData[1]) >= 529012:   # 20240120 动态
            # if 530887 >= float(eachData[1]) >= 530387:   # 20240120 动态  洪山广场
            # if 530362 >= float(eachData[1]) >= 530266:   # 20240120 动态  中南路
            # if 532239 >= float(eachData[1]) >= 532212:   # 20240120 动态  东湖凌波门
            # if 532239 >= float(eachData[1]) >= 532158:   # 20240120 动态  东湖南路
            # if 531408 >= float(eachData[1]) >= 531230:   # 20240120 动态  东三路
            # if 529895 >= float(eachData[1]) >= 529801:   # 20240120 动态  劝业场
            # if 536047 >= float(eachData[1]) >= 534035:   # 20240120 静态 星湖大楼东侧门口
            # if 529012 >= float(eachData[1]) >= 528000:   # 20240120 静态 星湖大楼西北角
            # if 529458 >= float(eachData[1]) >= 529020:   # 20240120 动态 星湖大楼四周绕圈 "02:55:00"-"03:04:00"
            # if 536020 >= float(eachData[1]) >= 533670:     # 20140120 动态 星湖大楼四周+东侧门口
            # if 534035 >= float(eachData[1]) >= 533671:     # 20140120 动态 星湖大楼四周
            if 536020 >= float(eachData[1]) >= 534035:     # 20140120 静态 星湖大楼东侧门口
                result.update(
                    {
                        time: {
                            "b": float(eachData[2]),
                            "l": float(eachData[3]),
                            "h": float(eachData[4]),
                            "num": float(eachData[9]),
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

def ReadPandaResult(_file):
    '''
    Author: hzh
    Date: 2025-01-25 23:34:34
    LastEditTime: 2025-01-26 10:12:12
    FilePath: /pyplot/com/panda2csv.py
    Descripttion: 
    '''
    result = {}
    with open(_file, "r") as file:
        lines = file.readlines()
        # 找到 "END OF HEADER" 行的索引
        end_header_index = next((i for i, line in enumerate(lines) if line.startswith("##-> END OF HEADER")), None)
        if end_header_index is None:
            raise ValueError("文件头部分未找到 '##-> END OF HEADER' 标记")
    
        content = lines[end_header_index + 1:]
        
        for eachLine in content:
            if eachLine.startswith("%"):
                continue
            # eachData = eachLine.split(",")
            # data.append(row)
            # GPSweek(int) + GPSsecond => week(float)
            eachData = eachLine.strip().split()
            ws = ymdhms2gpsws(
                int(eachData[0]),
                int(eachData[1]),
                int(eachData[2]),
                int(eachData[3]),
                int(eachData[4]),
                int(float(eachData[5])),
            )
            time = int(ws[0]) + float(ws[1]) / 86400.0 / 7.0
            # if 536020 >= float(eachData[1]) >= 533670:     # 20140120 动态 星湖大楼四周+东侧门口
            # if 534035 >= float(eachData[1]) >= 533671:     # 20140120 动态 星湖大楼四周
            # if 536020 >= float(eachData[1]) >= 534035:     # 20140120 静态 星湖大楼东侧门口
            result.update(
                {
                    time: {
                        "b": float(eachData[16]),
                        "l": float(eachData[17]),
                        "h": float(eachData[18]),
                        "num": float(eachData[19]),
                        "stat": float(eachData[26]),
                    }
                }
            )
    return result

if __name__ == '__main__':
    ppp = './data/202401/PPP/pos_2024020_ally_3d_if'
    stat = ReadPandaResult(ppp)
from math import isnan
import math
import sys
import os
from collections import namedtuple
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import pandas as pd
from mytime import ymdhms2gpsws


def read_posgo_pos(_file_):
    posgo = namedtuple("POSGO", " year moth day hour min sec lon lat height nsat pdop")

    if not os.path.isfile(_file_):
        print(_file_ + " is not exist.")
        sys.exit(1)

    neph = 0
    with open(_file_, "r") as fp:
        for line in fp:
            if line[0] == "%":
                continue
            neph = neph + 1

    pos = np.full((neph, 6 + 5), np.nan)

    ieph = -1
    with open(_file_, "r") as fp:
        for line in fp:
            if line[0] == "%":
                continue
            data = str(line).split(",")

            ymdhms = data[0].replace("/", " ")
            ymdhms = ymdhms.replace(":", " ")
            ymdhms = ymdhms[1 : len(ymdhms)].split(" ")

            gw, sow = ymdhms2gpsws(
                int(ymdhms[0]),
                int(ymdhms[1]),
                int(ymdhms[2]),
                int(ymdhms[3]),
                int(ymdhms[4]),
                int(float(ymdhms[5])),
            )

            ieph = ieph + 1

            pos[ieph][0] = int(ymdhms[0])
            pos[ieph][1] = int(ymdhms[1])
            pos[ieph][2] = int(ymdhms[2])
            pos[ieph][3] = int(ymdhms[3])
            pos[ieph][4] = int(ymdhms[4])
            pos[ieph][5] = float(ymdhms[5])

            pos[ieph][6] = float(data[1])  # B
            pos[ieph][7] = float(data[2])  # L
            pos[ieph][8] = float(data[3])  # H

            pos[ieph][9] = int(data[8])  # NSAT
            pos[ieph][10] = float(data[17])  # PDOP
            # print(pos[ieph][0],pos[ieph][1],pos[ieph][2],pos[ieph][3],pos[ieph][4],pos[ieph][5],pos[ieph][6],pos[ieph][7],pos[ieph][8],pos[ieph][9],pos[ieph][10],)

    return pos, neph


def plot_pos(POS, NEPH):
    dt = "5"
    Time_hms = []
    for ieph in range(0, NEPH):
        if math.isnan(POS[ieph][0]):
            continue
        Time_hms.append(
            "%04d-%02d-%02d %02d:%02d:%02d"
            % (
                POS[ieph][0],
                POS[ieph][1],
                POS[ieph][2],
                POS[ieph][3],
                POS[ieph][4],
                POS[ieph][5],
            )
        )
    Time_hms = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in Time_hms]
    myFmt = mdates.DateFormatter("%H:%M")

    plt.figure(dpi=300, figsize=(8, 5))

    plt.plot_date(
        Time_hms,
        POS[0 : len(Time_hms), 9],
        fmt="b+",
        markersize=2,
        label="{}".format(
            "AVG NSAT=" + str("%.3f" % (np.mean(POS[0 : len(Time_hms), 9])))
        ),
        alpha=0.9,
    )
    plt.plot_date(
        Time_hms,
        POS[0 : len(Time_hms) :, 10],
        fmt="r+",
        markersize=2,
        label="{}".format(
            "AVG PDOP=" + str("%.3f" % (np.mean(POS[0 : len(Time_hms), 10])))
        ),
        alpha=0.9,
    )
    plt.gca().set_xlim(Time_hms[0], Time_hms[-1])
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.xticks(pd.date_range(Time_hms[0], Time_hms[len(Time_hms) - 1], freq=dt + "T"))
    plt.ylabel("Number of Satellite")
    plt.xlabel("Time Span")
    plt.legend()
    plt.savefig("NSAT_PDOP.png")
    plt.show()

    return 0


def read_posgo_res(_file_, _prn_, _ifrq_):
    neph = 0
    with open(_file_, "r") as fp:
        for line in fp:
            if line[0] == "%":
                continue
            if line[0] == ">":
                neph = neph + 1

    res_info = np.full((neph, len(_prn_) + 6, 3), np.nan)

    ieph = -1
    with open(_file_, "r") as fp:
        for line in fp:
            if line[0] == "%":
                continue
            if line[0] == ">":
                gw, sow = ymdhms2gpsws(
                    int(line[1:5]),
                    int(line[6:8]),
                    int(line[9:11]),
                    int(line[12:14]),
                    int(line[15:17]),
                    int(float(line[18:24])),
                )
                # if sow < 130200 or sow > 131509:
                #     continue

                ieph = ieph + 1
                res_info[ieph, 0, :] = int(line[1:5])  # 年
                res_info[ieph, 1, :] = int(line[6:8])  # 月
                res_info[ieph, 2, :] = int(line[9:11])  # 日
                res_info[ieph, 3, :] = int(line[12:14])  # 时
                res_info[ieph, 4, :] = int(line[15:17])  # 分
                res_info[ieph, 5, :] = float(line[18:24])  # 秒
                # print(int(line[1:5]),int(line[6:8]),int(line[9:11]),int(line[12:14]),int(line[15:17]),float(line[18:24]))

            else:
                if int(line[11]) == _ifrq_:
                    for iprn in range(0, len(_prn_)):
                        if line[2:5] == _prn_[iprn]:
                            res_info[ieph][6 + iprn][0] = float(line[32:49])  # RES
                            if float(line[32:49]) == 0.0:
                                res_info[ieph][6 + iprn][0] = np.nan
                            res_info[ieph][6 + iprn][1] = float(line[87:105])  # ELE
                            res_info[ieph][6 + iprn][2] = float(line[107:119])  # SNR
                            # print(_prn_[iprn],res_info[ieph][6+iprn][0],res_info[ieph][6+iprn][1],res_info[ieph][6+iprn][2])

    return res_info, neph


def read_posgo_res_ai(_file_, _prn_, _ifrq_):
    neph = 0
    with open(_file_, "r") as fp:
        for line in fp:
            if line[0] == "%":
                continue
            if line[0] == ">":
                neph = neph + 1

    res_info = np.full((neph, len(_prn_) + 6, 3), np.nan)

    ieph = -1
    with open(_file_, "r") as fp:
        for line in fp:
            if line[0] == "%":
                continue
            if line[0] == ">":
                gw, sow = ymdhms2gpsws(
                    int(line[1:5]),
                    int(line[6:8]),
                    int(line[9:11]),
                    int(line[12:14]),
                    int(line[15:17]),
                    int(float(line[18:24])),
                )

                ieph = ieph + 1
                res_info[ieph, 0, :] = int(line[1:5])  # 年
                res_info[ieph, 1, :] = int(line[6:8])  # 月
                res_info[ieph, 2, :] = int(line[9:11])  # 日
                res_info[ieph, 3, :] = int(line[12:14])  # 时
                res_info[ieph, 4, :] = int(line[15:17])  # 分
                res_info[ieph, 5, :] = float(line[18:24])  # 秒
                # print(int(line[1:5]),int(line[6:8]),int(line[9:11]),int(line[12:14]),int(line[15:17]),float(line[18:24]))

            else:
                if int(line[11]) == _ifrq_:
                    for iprn in range(0, len(_prn_)):
                        if line[2:5] == _prn_[iprn]:
                            res_info[ieph][6 + iprn][0] = float(line[32:49])  # RES
                            if float(line[32:49]) == 0.0:
                                res_info[ieph][6 + iprn][0] = np.nan
                            res_info[ieph][6 + iprn][1] = float(line[87:105])  # ELE
                            res_info[ieph][6 + iprn][2] = float(line[107:119])  # SNR
                            # print(_prn_[iprn],res_info[ieph][6+iprn][0],res_info[ieph][6+iprn][1],res_info[ieph][6+iprn][2])

    return res_info, neph


def plot_res(RES, NEPH, PRN, TYPE):
    dt = "5"
    Time_hms = []
    for ieph in range(0, NEPH):
        if math.isnan(RES[ieph][0][0]):
            continue
        Time_hms.append(
            "%04d-%02d-%02d %02d:%02d:%02d"
            % (
                RES[ieph][0][0],
                RES[ieph][1][0],
                RES[ieph][2][0],
                RES[ieph][3][0],
                RES[ieph][4][0],
                RES[ieph][5][0],
            )
        )
    Time_hms = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in Time_hms]
    myFmt = mdates.DateFormatter("%H:%M")

    plt.figure(dpi=300, figsize=(8, 5))

    if TYPE == "res":
        allrms = 0
        alnum = 0
        for iprn in range(0, len(PRN)):
            nannum = 0
            rms = 0
            for j in range(0, len(RES[0 : len(Time_hms), 6 + iprn, 0])):
                if isnan(RES[j][6 + iprn][0]):
                    nannum = nannum + 1
                else:
                    rms = rms + RES[j][6 + iprn][0] ** 2
                    allrms = allrms + RES[j][6 + iprn][0] ** 2
            alnum = alnum + len(RES[0 : len(Time_hms), 6 + iprn, 0]) - nannum
            if nannum == len(RES[0 : len(Time_hms), 6 + iprn, 0]):
                continue
            rms = np.sqrt(rms / (len(RES[0 : len(Time_hms), 6 + iprn, 0]) - nannum))

            plt.plot_date(
                Time_hms,
                RES[0 : len(Time_hms), 6 + iprn, 0],
                fmt="+",
                markersize=2,
                label="{}".format(PRN[iprn] + " =" + str("%.3f" % (rms)), alpha=0.9),
            )

        allrms = np.sqrt(allrms / alnum)

        plt.ylim(-30, 30)
        plt.title(
            "Code Resdual, AVG_rms = " + str("%.3f" % (allrms)) + " m", fontsize=15
        )
        plt.gca().set_xlim(Time_hms[0], Time_hms[-1])
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.xticks(
            pd.date_range(Time_hms[0], Time_hms[len(Time_hms) - 1], freq=dt + "T")
        )
        plt.legend(ncol=3, fontsize=5)
        plt.ylabel("Code Resdual for SPP")
        plt.xlabel("Time Span")
        plt.savefig("CODE_res.png")
        plt.show()

    if TYPE == "snr":
        allmeans = 0
        alnum = 0

        for iprn in range(0, len(PRN)):
            nannum = 0
            mean = 0
            for j in range(0, len(RES[0 : len(Time_hms), 6 + iprn, 2])):
                if isnan(RES[j][6 + iprn][2]):
                    nannum = nannum + 1
                else:
                    mean = mean + RES[j][6 + iprn][2]
                    allmeans = allmeans + RES[j][6 + iprn][2]
            alnum = alnum + len(RES[0 : len(Time_hms), 6 + iprn, 2]) - nannum
            if nannum == len(RES[0 : len(Time_hms), 6 + iprn, 2]):
                continue
            mean = mean / (len(RES[0 : len(Time_hms), 6 + iprn, 2]) - nannum)

            plt.plot_date(
                Time_hms,
                RES[0 : len(Time_hms), 6 + iprn, 2],
                fmt="+",
                markersize=2,
                label="{}".format(PRN[iprn] + " =" + str("%.2f" % (mean)), alpha=0.9),
            )

        allmeans = allmeans / alnum

        plt.ylim(20, 60)
        plt.title("SNR, AVG_snr = " + str("%.2f" % (allmeans)), fontsize=15)
        plt.gca().set_xlim(Time_hms[0], Time_hms[-1])
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.xticks(
            pd.date_range(Time_hms[0], Time_hms[len(Time_hms) - 1], freq=dt + "T")
        )
        plt.legend(ncol=3, fontsize=5)
        plt.ylabel("SNR [dB]")
        plt.xlabel("Time Span")
        plt.savefig("SNR.png")
        plt.show()

    if TYPE == "snr_ele":
        allmeans = 0
        alnum = 0

        for iprn in range(0, len(PRN)):
            nannum = 0
            mean = 0
            for j in range(0, len(RES[0 : len(Time_hms), 6 + iprn, 1])):
                if isnan(RES[j][6 + iprn][1]):
                    nannum = nannum + 1
                else:
                    mean = mean + RES[j][6 + iprn][1]
                    allmeans = allmeans + RES[j][6 + iprn][1]
            alnum = alnum + len(RES[0 : len(Time_hms), 6 + iprn, 1]) - nannum
            if nannum == len(RES[0 : len(Time_hms), 6 + iprn, 1]):
                continue
            mean = mean / (len(RES[0 : len(Time_hms), 6 + iprn, 1]) - nannum)
            plt.scatter(
                RES[0 : len(Time_hms), 6 + iprn, 1],
                RES[0 : len(Time_hms), 6 + iprn, 2],
                marker="+",
                label=PRN[iprn],
                s=3,
            )

        allmeans = allmeans / alnum
        plt.ylabel
        plt.ylim(20, 60)
        plt.title("SNR_vs_ELE, AVG_ele = " + str("%.2f" % (allmeans)), fontsize=15)
        plt.legend(ncol=3, fontsize=5)
        plt.ylabel("SNR [dB]")
        plt.xlabel("Elevation [degree]")
        plt.savefig("SNR_ELE.png")
        plt.show()

    return 0


def pos2kml(POS, NEPH, outname):
    outkml = open(outname, "w")
    print("Longitude,Latitude,Elevation,name", file=outkml)

    for ieph in range(0, NEPH):
        if (
            math.isnan(POS[ieph][3])
            or math.isnan(POS[ieph][4])
            or math.isnan(POS[ieph][5])
        ):
            continue
        hour = str(int(POS[ieph][3]))
        if int(POS[ieph][3]) < 10:
            hour = "0" + str(int(POS[ieph][3]))

        min = str(int(POS[ieph][4]))
        if int(POS[ieph][4]) < 10:
            min = "0" + str(int(POS[ieph][4]))

        sec = str(int(POS[ieph][5]))
        if int(POS[ieph][5]) < 10:
            sec = "0" + str(int(POS[ieph][5]))

        msg = "%.9f, %.9f, %.9f, %2s:%2s:%2s" % (
            POS[ieph][7],
            POS[ieph][6],
            POS[ieph][8],
            hour,
            min,
            sec,
        )
        print(msg, file=outkml)


if __name__ == "__main__":
    prn = [
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
        "G06",
        "G07",
        "G08",
        "G09",
        "G10",
        "G11",
        "G12",
        "G13",
        "G14",
        "G15",
        "G16",
        "G17",
        "G18",
        "G19",
        "G20",
        "G21",
        "G22",
        "G23",
        "G24",
        "G25",
        "G26",
        "G27",
        "G28",
        "G29",
        "G30",
        "G31",
        "G32",
        "R01",
        "R02",
        "R03",
        "R04",
        "R05",
        "R06",
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R12",
        "R13",
        "R14",
        "R15",
        "R16",
        "R17",
        "R18",
        "R19",
        "R20",
        "R21",
        "R22",
        "R23",
        "R24",
        "R25",
        "R26",
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C35",
        "C36",
        "C37",
        "C38",
        "C39",
        "C40",
        "C41",
        "C42",
        "C43",
        "C44",
        "C45",
        "C46",
        "C47",
        "C48",
        "C49",
        "C50",
        "C51",
        "C52",
        "C53",
        "C54",
        "C55",
        "C56",
        "C57",
        "C58",
        "C59",
        "E01",
        "E02",
        "E03",
        "E04",
        "E05",
        "E06",
        "E07",
        "E08",
        "E09",
        "E10",
        "E11",
        "E12",
        "E13",
        "E14",
        "E15",
        "E16",
        "E17",
        "E18",
        "E19",
        "E20",
        "E21",
        "E22",
        "E23",
        "E24",
        "E25",
        "E26",
        "E27",
        "E28",
        "E29",
        "E30",
        "E31",
        "E32",
        "E33",
        "E34",
        "E35",
        "E36",
        "J01",
        "J02",
        "J03",
        "J04",
        "J05",
        "J06",
        "J07",
    ]

    prnc = [
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C35",
        "C36",
        "C37",
        "C38",
        "C39",
        "C40",
        "C41",
        "C42",
        "C43",
        "C44",
        "C45",
        "C46",
        "C47",
        "C48",
        "C49",
        "C50",
        "C51",
        "C52",
        "C53",
        "C54",
        "C55",
        "C56",
        "C57",
        "C58",
        "C59",
    ]

    prng = [
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
        "G06",
        "G07",
        "G08",
        "G09",
        "G10",
        "G11",
        "G12",
        "G13",
        "G14",
        "G15",
        "G16",
        "G17",
        "G18",
        "G19",
        "G20",
        "G21",
        "G22",
        "G23",
        "G24",
        "G25",
        "G26",
        "G27",
        "G28",
        "G29",
        "G30",
        "G31",
        "G32",
    ]

    prn = prnc + prng

    mode = sys.argv[1]

    if mode == "pos":
        # 绘制NSAT和PDOP
        pos_file = sys.argv[2]
        POS, NEPH = read_posgo_pos(pos_file)
        plot_pos(POS, NEPH)

    if mode == "res":
        # 绘制残差和信噪比
        res_file = sys.argv[2]
        RES, NEPH = read_posgo_res_ai(res_file, prn, 1)

        type = "snr_ele"  # snr / res / snr_ele
        plot_res(RES, NEPH, prn, type)

        type = "snr"  # snr / res / snr_ele
        plot_res(RES, NEPH, prn, type)

        type = "res"  # snr / res / snr_ele
        plot_res(RES, NEPH, prn, type)

    if mode == "pos2kml":
        pos_file = sys.argv[2]
        outname = sys.argv[3]
        POS, NEPH = read_posgo_pos(pos_file)
        pos2kml(POS, NEPH, outname)

    print("Over !")

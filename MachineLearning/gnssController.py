"""
Author: hzh huzihe@whu.edu.cn
Date: 2023-08-16 20:40:22
LastEditTime: 2023-08-19 20:19:27
FilePath: /pyplot/MachineLearning/gnssController.py
Descripttion: 
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from com.mytime import ymdhms2gpsws
from com.mystr import replace_char

import gnss_xgboost
import gnss_gmm
import gnss_kmeans
import gnss_fcm
import gnss_svm


def out_gnss_data(rnx, outrnx, satInfo, istrimble):
    result = {}
    with open(rnx, "r") as file:
        content = file.readlines()
        headerflag = False
        with open(outrnx, "w") as outfile:
            for eachLine in content:
                if eachLine.startswith("%"):  # ignore comment line
                    outfile.write(eachLine)  # 输出 rnx 文件头
                    continue
                ender = "END OF HEADER" in eachLine
                if ender:
                    headerflag = True
                    outfile.write(eachLine)  # 输出 rnx 文件头
                    continue
                elif headerflag:
                    if eachLine.startswith(">"):
                        outfile.write(eachLine)  # 输出时间部分

                        eachData = eachLine.split()
                        ws = ymdhms2gpsws(
                            int(eachData[1]),
                            int(eachData[2]),
                            int(eachData[3]),
                            int(eachData[4]),
                            int(eachData[5]),
                            int(float(eachData[6])),
                        )
                        continue
                    else:
                        eachData = eachLine.split()
                        los = satInfo.loc[
                            (satInfo["week"] == ws[0])
                            & (satInfo["second"] == ws[1])
                            & (satInfo["sat"] == eachData[0]),
                            :,
                        ]
                        if los.size <= 0:
                            outfile.write(eachLine)
                        else:
                            nlosflag = str(los.iat[0, 3])
                            for itr in range(6):
                                if len(eachLine) > 48 * itr + 18 + 16:
                                    substr = eachLine[48 * itr + 18]
                                    if substr != " ":
                                        eachLine = replace_char(
                                            eachLine, nlosflag, 48 * itr + 18
                                        )
                                        eachLine = replace_char(
                                            eachLine, nlosflag, 48 * itr + 18 + 16
                                        )
                                        if not istrimble:
                                            eachLine = replace_char(
                                                eachLine, nlosflag, 48 * itr + 18 + 32
                                            )
                                            eachLine = replace_char(
                                                eachLine, nlosflag, 48 * itr + 18 + 48
                                            )
                            outfile.write(eachLine)  # 输出nlos修订后的obs记录
                        continue
                else:
                    outfile.write(eachLine)  # 输出 rnx 文件头
                    continue


if __name__ == "__main__":
    # resouce path
    path1 = "./data/ml-data/trimble.res1"
    path2 = "./data/ml-data/X6833B.res1"

    # Supervised learning model path
    xgboost_trimble_modelpath = "./data/ml-data/gnss_xgboost_trimble.model"
    xgboost_X6833B_modelpath = "./data/ml-data/gnss_xgboost_X6833B.model"
    gmm_X6833B_modelpath = "./data/ml-data/gnss_gmm_X6833B.model"
    kmeans_X6833B_modelpath = "./data/ml-data/gnss_kmeans_X6833B.model"
    fcm_X6833B_modelpath = "./data/ml-data/gnss_fcm_X6833B.model"
    svm_X6833B_modelpath = "./data/ml-data/gnss_svm_X6833B.model"

    # model traning
    # xgboost_gnss.xgboost_gnss_train_model(path1, modelpath1)
    gnss_xgboost.xgboost_gnss_train_model(path2, xgboost_X6833B_modelpath)
    gnss_gmm.gnss_gmm_train_model(path2, gmm_X6833B_modelpath)
    gnss_kmeans.gnss_kmeans_train_model(path2, kmeans_X6833B_modelpath)
    gnss_fcm.gnss_fcm_train_model(path2, fcm_X6833B_modelpath)
    gnss_svm.gnss_svm_train_model(path2, svm_X6833B_modelpath)

    # predict
    # satinfo_ref = xgboost_gnss.xgboost_gnss_predict(modelpath1, path1)
    satinfo_ref = gnss_xgboost.xgboost_gnss_predict(xgboost_X6833B_modelpath, path2)
    # satinfo_ref = xgboost_gnss.xgboost_gnss_predict(modelpath1, path2)
    # satinfo_ref = xgboost_gnss.xgboost_gnss_predict(modelpath2, path1)

    # mark los/nlos for rnx data by un/supervised learning
    # rnx = "./data/ml-data/trimble-3dma-0520.rnx"
    # outrnx = "./data/ml-data/trimble-3dma-0520-ai.rnx"
    # out_gnss_data(rnx, outrnx, satinfo_ref, 1)
    rnx = "./data/ml-data/X6833B-3dma-0730.rnx"
    outrnx = "./data/ml-data/X6833B-3dma-0730-ai.rnx"
    out_gnss_data(rnx, outrnx, satinfo_ref, 0)

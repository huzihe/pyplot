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
                                    substr = eachLine[48 * itr + 16]
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
                                            if len(eachLine) >= 48 * itr + 18 + 48:
                                                if eachLine[48 * itr + 18 + 48] != " ":
                                                    eachLine = replace_char(
                                                        eachLine,
                                                        nlosflag,
                                                        48 * itr + 18 + 48,
                                                    )
                            outfile.write(eachLine)  # 输出nlos修订后的obs记录
                        continue
                else:
                    outfile.write(eachLine)  # 输出 rnx 文件头
                    continue


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("\33[1;31m**Usage \33[0m")
    #     sys.exit(1)
    # else:
    #     # 1：重新训练并保存模型；0：不训练模型
    #     istrainmodel = sys.argv[0]
    #     learningtype = sys.argv[1]

    istrainmodel = 0
    learningtype = "kmeans"

    # resouce path
    # X6833B  对应前39091行为静态数据，后12670（51761-39091）行为动态数据
    # trimble 对应前46082行为静态数据，后9873（55955-46082）行为动态数据
    # ublox 对应33405行为静态数据，后12978（46383-33405）行为动态数据
    # CK6n 对应33011行为静态数据，后9579（42590-33011）行为动态数据
    trimble_path = "./data/ml-data/20230511/trimble.res1"
    X6833B_path = "./data/ml-data/20230511/X6833B.res1"
    ublox_path = "./data/ml-data/20230511/ublox.res1"
    CK6n_path = "./data/ml-data/20230511/CK6n.res1"
    X6833B_full_path = "./data/ml-data/20230511/X6833B-full-1001.res1"
    CK6n_full_path = "./data/ml-data/20230511/CK6n-lsq-full-predata-1005.res1"
    ublox_full_path = "./data/ml-data/20230511/ublox-spp-lsq-1006.res1"
    trimble_full_path = "./data/ml-data/20230511/Trimble-spp-lsq-1006.res1"


    if learningtype == "xgboost":
        # Supervised learning model path
        trimble_modelpath = "./data/ml-data/model/gnss_xgboost_trimble.model"
        X6833B_modelpath = "./data/ml-data/model/gnss_xgboost_X6833B.model"
        ublox_modelpath = "./data/ml-data/model/gnss_xgboost_ublox.model"
        CK6n_modelpath = "./data/ml-data/model/gnss_xgboost_CK6n.model"

        # model traning
        if istrainmodel:
            gnss_xgboost.xgboost_gnss_train_model(trimble_path, trimble_modelpath)
            gnss_xgboost.xgboost_gnss_train_model(X6833B_path, X6833B_modelpath)
            gnss_xgboost.xgboost_gnss_train_model(ublox_path, ublox_modelpath)
            gnss_xgboost.xgboost_gnss_train_model(CK6n_path, CK6n_modelpath)

        # predict
        # satinfo_t = gnss_xgboost.xgboost_gnss_predict(trimble_modelpath, trimble_path)
        # satinfo_X = gnss_xgboost.xgboost_gnss_predict(X6833B_modelpath, X6833B_path)
        # satinfo_ublox = gnss_xgboost.xgboost_gnss_predict(ublox_modelpath, ublox_path)
        # satinfo_CK6n = gnss_xgboost.xgboost_gnss_predict(CK6n_modelpath, CK6n_path)

        # # mark los/nlos for rnx data by un/supervised learning
        # rnx = "./data/ml-data/20230511/trimble-3dma-0520.rnx"
        # outrnx = "./data/ml-data/20230511/trimble-3dma-0520-xgboost.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_t, 1)

        # rnx = "./data/ml-data/20230511/X6833B-3dma-0730.rnx"
        # outrnx = "./data/ml-data/20230511/X6833B-3dma-0730-xgboost.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 0)

        # rnx = "./data/ml-data/20230511/ublox-3dma-0730.rnx"
        # outrnx = "./data/ml-data/20230511/ublox-3dma-0730-xgboost.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_ublox, 1)

        # rnx = "./data/ml-data/20230511/CK6n-3dma-0730.rnx"
        # outrnx = "./data/ml-data/20230511/CK6n-3dma-0730-xgboost.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_CK6n, 0)
        # satinfo_X = gnss_xgboost.xgboost_gnss_predict(X6833B_modelpath, X6833B_full_path)

        # rnx = "./data/ml-data/20230511/X6833B.23o"
        # outrnx = "./data/ml-data/20230511/X6833B-3dma-xgboost-1001.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 0)

        # satinfo_X = gnss_xgboost.xgboost_gnss_predict(CK6n_modelpath, CK6n_full_path)
        # rnx = "./data/ml-data/20230511/CK6n.23o"
        # outrnx = "./data/ml-data/20230511/CK6n-xgboost-1005.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 1)

        satinfo_X = gnss_xgboost.xgboost_gnss_predict(trimble_modelpath, trimble_full_path)
        rnx = "./data/ml-data/20230511/IGS000USA_R_20231310150_01D_01S_MO.rnx"
        outrnx = "./data/ml-data/20230511/trimble-xgboost-1006.rnx"
        out_gnss_data(rnx, outrnx, satinfo_X, 1)

        # satinfo_X = gnss_xgboost.xgboost_gnss_predict(ublox_modelpath, ublox_full_path)
        # rnx = "./data/ml-data/20230511/ublox.obs"
        # outrnx = "./data/ml-data/20230511/ulbox-xgboost-1006.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 0)

    elif learningtype == "svm":
        svm_trimble_modelpath = "./data/ml-data/model/gnss_svm_trimble.model"
        svm_X6833B_modelpath = "./data/ml-data/model/gnss_svm_X6833B.model"
        svm_ublox_modelpath = "./data/ml-data/model/gnss_svm_ublox.model"
        svm_CK6n_modelpath = "./data/ml-data/model/gnss_svm_CK6n.model"

        # model traning
        if istrainmodel:
            gnss_svm.gnss_svm_train_model(trimble_path, svm_trimble_modelpath)
            gnss_svm.gnss_svm_train_model(X6833B_path, svm_X6833B_modelpath)
            gnss_svm.gnss_svm_train_model(ublox_path, svm_ublox_modelpath)
            gnss_svm.gnss_svm_train_model(CK6n_path, svm_CK6n_modelpath)

        # predict
        # satinfo = gnss_svm.gnss_svm_predict(svm_trimble_modelpath, trimble_path)
        # satinfo = gnss_svm.gnss_svm_predict(svm_X6833B_modelpath, X6833B_path)
        # satinfo = gnss_svm.gnss_svm_predict(svm_ublox_modelpath, ublox_path)
        # satinfo = gnss_svm.gnss_svm_predict(svm_CK6n_modelpath, CK6n_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_trimble_modelpath, trimble_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_trimble_modelpath, X6833B_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_trimble_modelpath, ublox_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_trimble_modelpath, CK6n_path)

        satinfo = gnss_svm.gnss_svm_predict(svm_X6833B_modelpath, trimble_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_X6833B_modelpath, ublox_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_X6833B_modelpath, CK6n_path)

        satinfo = gnss_svm.gnss_svm_predict(svm_ublox_modelpath, trimble_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_ublox_modelpath, X6833B_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_ublox_modelpath, CK6n_path)

        satinfo = gnss_svm.gnss_svm_predict(svm_CK6n_modelpath, trimble_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_CK6n_modelpath, X6833B_path)
        satinfo = gnss_svm.gnss_svm_predict(svm_CK6n_modelpath, ublox_path)

    elif learningtype == "gmm":
        gmm_X6833B_modelpath = "./data/ml-data/model/gnss_gmm_X6833B.model"

        # model traning
        if istrainmodel:
            gnss_gmm.gnss_gmm_train_model(X6833B_path, gmm_X6833B_modelpath)
        # predict
        satinfo = gnss_gmm.gnss_gmm_predict(gmm_X6833B_modelpath, X6833B_path)

    elif learningtype == "kmeans":
        trimble_modelpath = "./data/ml-data/model/gnss_kmeans_trimble.model"
        X6833B_modelpath = "./data/ml-data/model/gnss_kmeans_X6833B.model"
        ublox_modelpath = "./data/ml-data/model/gnss_kmeans_ublox.model"
        CK6n_modelpath = "./data/ml-data/model/gnss_kmeans_CK6n.model"

        # model traning
        if istrainmodel:
            gnss_kmeans.gnss_kmeans_train_model(trimble_path, trimble_modelpath)
            gnss_kmeans.gnss_kmeans_train_model(X6833B_path, X6833B_modelpath)
            gnss_kmeans.gnss_kmeans_train_model(ublox_path, ublox_modelpath)
            gnss_kmeans.gnss_kmeans_train_model(CK6n_path, CK6n_modelpath)

        # predict
        # satinfo_t = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, trimble_path)
        # satinfo_X = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, X6833B_path)
        # satinfo_ublox = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, ublox_path)
        # satinfo_CK6n = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, CK6n_path)

        # # mark los/nlos for rnx data by un/supervised learning
        # rnx = "./data/ml-data/20230511/trimble-3dma-0520.rnx"
        # outrnx = "./data/ml-data/20230511/trimble-3dma-0520-kmeans.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_t, 1)

        # rnx = "./data/ml-data/20230511/X6833B-3dma-0730.rnx"
        # outrnx = "./data/ml-data/20230511/X6833B-3dma-0730-kmeans.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 0)

        # rnx = "./data/ml-data/20230511/ublox-3dma-0730.rnx"
        # outrnx = "./data/ml-data/20230511/ublox-3dma-0730-kmeans.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_ublox, 1)
        # satinfo_X = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, X6833B_full_path)

        # rnx = "./data/ml-data/20230511/X6833B.23o"
        # outrnx = "./data/ml-data/20230511/X6833B-3dma-kmeans-1001.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 0)
        
        # satinfo_X = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, CK6n_full_path)
        # rnx = "./data/ml-data/20230511/CK6n.23o"
        # outrnx = "./data/ml-data/20230511/CK6n-kmeans-1005.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 0)

        satinfo_X = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, ublox_full_path)
        rnx = "./data/ml-data/20230511/ublox.obs"
        outrnx = "./data/ml-data/20230511/ublox-kmeans-1006.rnx"
        out_gnss_data(rnx, outrnx, satinfo_X, 0)

        # satinfo_X = gnss_kmeans.gnss_kmeans_predict(X6833B_modelpath, trimble_full_path)
        # rnx = "./data/ml-data/20230511/IGS000USA_R_20231310150_01D_01S_MO.rnx"
        # outrnx = "./data/ml-data/20230511/trimble-kmeans-1006.rnx"
        # out_gnss_data(rnx, outrnx, satinfo_X, 1)

    elif learningtype == "fcm":
        fcm_trimble_modelpath = "./data/ml-data/model/gnss_fcm_trimble.model"
        fcm_X6833B_modelpath = "./data/ml-data/model/gnss_fcm_X6833B.model"

        # model traning
        if istrainmodel:
            gnss_fcm.gnss_fcm_train_model(X6833B_path, fcm_X6833B_modelpath)

        # predict
        satinfo = gnss_fcm.gnss_fcm_predict(fcm_X6833B_modelpath, X6833B_path)
        satinfo = gnss_fcm.gnss_fcm_predict(fcm_trimble_modelpath, X6833B_path)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from com.sqlite3d import SqliteSearch

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
# plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 9  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8

def DrawSkyPlot(satNO,sataz,satel,el,bgurl,filepath,symbleflag):
    az = []
    index = 0
    for item in el:
        az.append((360 / len(el)) * index * np.pi / 180)
        index += 1
    az.append(2 * np.pi)
    el.append(el[0])

    inch = 1/2.54
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, dpi=300, figsize=(8.4 * inch, 8.4 * inch))

    # # 绘制背景图片
    if os.path.exists(bgurl):
        img = plt.imread(bgurl)
        if img is not None:
            axes_coords = [0.123, 0.111, 0.779, 0.77] # plotting full width and height
            ax_image = fig.add_axes(axes_coords, label="ax image")
            ax_image.imshow(img, alpha=.4)
            ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image
        else:
            print("图片读取失败！")
    else:
        print("文件路径不存在！")

    # 天空阴影图极坐标绘制
    ax.patch.set_facecolor("0.9")  # 底色设置为灰色
    ax.plot(az, el, linewidth=1.0, color='C9')  # 绘制建筑边界
    ax.fill(az, el, "w")  # 中间天空填充白色

    # # 绘制卫星，用不同形状表示不同星座
    if len(sataz)>0 and len(satel)>0 and len(satNO)>0:
        for x, y, t in zip(sataz, satel, satNO):
            s = t[0]
            if symbleflag:
                if s == "G":
                    ax.plot(x, y, marker="^", color="g", markersize=4)
                elif s == "R":
                    ax.plot(x, y, marker="s", color="m", markersize=4)
                elif s == "E":
                    ax.plot(x, y, marker="d", color="b", markersize=4)
                elif s == "J":
                    ax.plot(x, y, marker="o", color="y", markersize=4)
                elif s == "C":
                    ax.plot(x, y, marker="p", color="r", markersize=4)
                else:
                    ax.plot(x, y, "ro")
            else:
                ax.plot(x, y, marker="o", color="b", markersize=4)
            ax.text(
                x,
                y,
                t,
                horizontalalignment="left",
                verticalalignment="bottom",
                color="darkslategray",
                fontsize=7,
            )
    else:
        print("注意：sataz satel satNO 数组存在为空！")
    # 绘制卫星结束

    ax.set_rmax(2)
    ax.set_rticks([90, 80, 60, 40, 20])  # Less radial ticks
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_theta_zero_location("N")  # 0°位置为正北方向
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_theta_direction(-1)  # 顺时针
    ax.set_rlim(90, 0)
    # ax.grid(True)

    # ax.set_title("A skymask on a polar axis", va='top')
    # plt.savefig("./data/polaraxis-0120-P40.png")  # 保存图片
    plt.savefig(filepath)
    plt.show()

def DrawSkyPlotNLOS(satNO,nlosflag,sataz,satel,el,bgurl,filepath):
    az = []
    index = 0
    for item in el:
        az.append((360 / len(el)) * index * np.pi / 180)
        index += 1
    az.append(2 * np.pi)
    el.append(el[0])

    inch = 1/2.54
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, dpi=300, figsize=(8.4 * inch, 8.4 * inch))

    # # 绘制背景图片
    if os.path.exists(bgurl):
        img = plt.imread(bgurl)
        if img is not None:
            axes_coords = [0.123, 0.111, 0.779, 0.77] # plotting full width and height
            ax_image = fig.add_axes(axes_coords, label="ax image")
            ax_image.imshow(img, alpha=.4)
            ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image
        else:
            print("图片读取失败！")
    else:
        print("文件路径不存在！")

    # 天空阴影图极坐标绘制
    ax.patch.set_facecolor("0.9")  # 底色设置为灰色
    ax.plot(az, el, linewidth=1.0, color='C9')  # 绘制建筑边界
    ax.fill(az, el, "w")  # 中间天空填充白色

    # # 绘制卫星，用不同形状表示不同星座
    if len(sataz)>0 and len(satel)>0 and len(satNO)>0:
        for x, y, t, f in zip(sataz, satel, satNO, nlosflag):
            if f==0:
                ax.plot(x,y,marker="o", color="r", markersize=4)
            else:
                ax.plot(x,y,marker="o", color="b", markersize=4)
            ax.text(
                x,
                y,
                t,
                horizontalalignment="left",
                verticalalignment="bottom",
                color="darkslategray",
                fontsize=7,
            )
    else:
        print("注意：sataz satel satNO 数组存在为空！")
    # 绘制卫星结束

    ax.set_rmax(2)
    ax.set_rticks([80, 60, 40, 20])  # Less radial ticks
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_theta_zero_location("N")  # 0°位置为正北方向
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_theta_direction(-1)  # 顺时针
    ax.set_rlim(90, 0)
    # ax.grid(True)

    # ax.set_title("A skymask on a polar axis", va='top')
    # plt.savefig("./data/polaraxis-0120-P40-dongmen.png")  # 保存图片
    plt.savefig(filepath)
    plt.show()

def skytest():

    theta=np.arange(0,2*np.pi,0.02)
    ax1= plt.subplot(121, projection='polar')
    ax2= plt.subplot(122, projection='polar')
    ax2.set_rlabel_position('90')
    ax1.plot(theta,theta/6,'--',lw=2)
    ax2.plot(theta,theta/6,'--',lw=2)
    plt.show()


if __name__ == '__main__':

    # 数据准备
    # # 中北路 PT1 Trimble
    satNO=['G08','G21','G30','R12','R21','R22','R23','E24','E25','C03','C07','C09','C10','C26','C35','C40','C45',]
    sataz=[0.460737,2.242701,5.474722,0.013700,0.492199,0.347244,3.669284,0.403945,5.466504,3.270222,0.321267,3.735483,5.575213,5.565875,4.100780,6.049571,1.780578,]
    satel=[54.069951,62.056097,33.376186,58.847369,27.679740,79.820890,29.450760,65.651789,28.059449,53.025364,66.306727,36.955029,56.718201,40.172611,67.757415,57.084646,79.008884,]
    el=[59,60,60,61,61,61,61,61,59,57,56,57,20,20,20,20,26,61,65,68,70,70,70,70,70,69,69,68,68,76,77,77,78,78,78,79,79,79,79,80,80,80,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,82,81,81,81,81,81,81,81,81,81,81,81,81,81,81,81,80,80,80,79,79,79,79,78,78,78,59,57,56,55,55,61,61,61,61,62,62,63,61,58,53,43,43,43,43,43,43,42,42,28,39,61,61,61,61,61,60,59,59,59,58,58,57,57,56,23,24,25,26,26,27,28,53,54,54,55,55,79,79,79,79,79,79,79,80,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,82,82,82,82,82,81,81,39,49,50,52,53,52,52,51,50,53,54,54,55,55,54,]

    # # 中北路 PT2-02:30 Trimble
    # satNO=['G01','G07','G08','G14','G17','G21','G30','R12','R13','R22','R23','E11','E24','E25','C02','C05','C07','C10','C26','C40','C44','C56','C60',]
    # sataz=[2.569071,3.782189,0.759838,5.435713,4.501962,1.229513,5.141663,1.237173,5.819112,0.309242,4.108419,3.402881,1.291107,5.646663,3.968185,4.355162,0.592044,5.723580,5.426667,6.261490,4.208554,0.000000,4.097716,]
    # satel=[68.454162,63.673920,35.979609,26.822518,23.101683,64.460845,50.098043,76.844966,36.790393,50.792087,52.288991,68.495318,68.226596,42.207982,39.964458,19.992633,68.546950,59.558022,59.294015,59.006897,30.455750,0.000000,39.649200,]
    # el=[59,60,60,59,58,58,58,44,44,44,42,41,17,17,17,50,54,54,58,58,58,57,57,57,74,75,76,77,78,78,78,79,79,79,80,80,81,81,81,81,81,82,82,82,82,82,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,82,82,82,70,70,69,68,68,68,67,66,66,65,64,63,62,61,59,60,61,61,61,61,62,62,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,62,52,55,57,59,59,58,57,57,57,56,55,55,54,53,52,30,31,31,30,29,28,26,25,24,28,28,79,79,79,79,79,79,79,79,79,79,83,83,83,83,83,83,82,82,82,82,82,82,82,82,81,81,81,81,81,27,26,26,33,35,36,37,36,37,56,57,57,58,59,59,]

    # # 中北路 PT2-02:45 Trimble
    # satNO=['G01','G14','G17','G28','G30','R12','R13','R23','E11','E12','E25','J03','C05','C07','C10','C29','C35','C38','C40','C44','C56','C60',]
    # sataz=[2.243071,5.505826,4.613748,5.389198,4.960014,1.894925,5.836325,4.336082,3.500378,0.729431,5.686373,1.082890,4.355507,0.688637,5.758287,0.739402,5.865306,3.101360,0.041020,4.311886,0.000000,4.100038,]
    # satel=[74.112540,31.959739,27.067324,18.213274,53.116695,74.933814,44.447347,57.681923,74.873105,46.400750,47.015853,68.126224,20.012517,69.508839,60.782903,30.752292,73.497651,43.935787,60.084294,34.820462,0.000000,39.750835,]
    # el=[59,60,60,59,58,58,58,44,44,44,42,41,17,17,17,50,54,54,58,58,58,57,57,57,74,75,76,77,78,78,78,79,79,79,80,80,81,81,81,81,81,82,82,82,82,82,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,83,82,82,82,70,70,69,68,68,68,67,66,66,65,64,63,62,61,59,60,61,61,61,61,62,62,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,62,52,55,57,59,59,58,57,57,57,56,55,55,54,53,52,30,31,31,30,29,28,26,25,24,28,28,79,79,79,79,79,79,79,79,79,79,83,83,83,83,83,83,82,82,82,82,82,82,82,82,81,81,81,81,81,27,26,26,33,35,36,37,36,37,56,57,57,58,59,59,]

    # # 中北路 PT3 Trimble
    # satNO=['G07','G08','G14','G21','G30','R13','R22','R23','E11','E25','J03','C07','C08','C10','C26','C29','C35','C38','C40','C44','C56','C60',]
    # sataz=[3.489546,0.959806,5.588877,0.850560,4.687532,5.823840,0.424082,4.713104,3.797481,5.722665,1.030835,0.827014,3.016348,5.794344,4.980221,0.801208,6.214926,3.158572,0.121287,4.460301,0.000000,4.102804,]
    # satel=[48.352033,25.558051,38.920169,53.260526,54.531118,54.629584,33.688699,62.290141,82.554957,53.466485,67.470770,70.836956,53.630603,62.581096,69.443081,24.519960,68.361813,48.365294,61.776020,39.963160,0.000000,39.870641,]
    # el=[22,22,25,25,26,26,26,25,23,23,23,23,26,26,26,26,17,38,45,51,54,54,54,53,53,52,52,49,85,85,85,85,86,86,86,86,86,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,87,86,86,86,86,86,85,85,85,85,85,85,84,84,83,83,82,81,80,79,78,45,45,46,46,42,28,27,27,24,33,33,46,48,48,48,47,47,46,46,46,45,34,33,76,76,83,83,83,83,83,83,84,84,84,84,84,85,85,85,85,85,85,84,83,82,81,78,74,67,7,7,7,25,30,30,7,9,11,12,13,15,17,18,19,20,21,22,24,24,26,26,28,28,29,30,31,31,31,30,28,26,25,24,]

    # # 吉庆街 PT1 Trimble
    # satNO=['G10','G18','G23','G24','R02','R03','R17','E03','E13','E15','E24','E25','J01','J02','J07','C01','C02','C03','C04','C06','C08','C09','C13','C16','C21','C22','C36','C38','C39','C44','C45','C59','C60',]
    # sataz=[5.496433,3.885300,5.943939,1.558042,0.498319,5.542911,3.914799,6.139322,5.502687,4.488118,1.012755,1.973932,1.106051,2.341624,2.722928,2.290122,4.021355,3.270193,2.038772,4.671350,0.000000,4.367573,4.113594,5.056281,5.385629,0.480419,1.467668,2.780908,5.634608,4.144515,2.831476,2.376285,4.054193,]
    # satel=[27.784440,62.911899,58.407927,72.506624,42.132825,34.663143,32.646976,61.465135,23.055149,54.995783,14.058676,25.963460,59.127584,30.325341,51.792355,41.445369,42.622521,54.895482,28.902784,68.293282,0.000000,57.399464,82.863720,73.680050,35.271161,58.622685,46.328445,60.910478,73.927460,26.157056,39.325585,45.288651,37.783510,]
    # el = [7,16,18,18,19,20,22,24,27,30,31,34,36,38,40,41,64,65,66,67,68,68,69,70,70,71,72,72,72,73,73,73,72,70,68,66,65,61,55,50,44,13,12,7,7,7,7,7,7,7,7,7,7,34,37,40,42,45,47,49,51,53,54,55,57,57,56,56,57,58,59,59,60,61,61,62,62,63,63,63,64,64,64,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,64,64,68,68,68,68,68,68,68,68,68,68,68,68,68,67,67,67,66,66,65,63,62,60,58,22,22,10,10,9,9,9,28,28,29,30,30,29,26,18,18,18,18,26,26,26,26,21,22,24,26,27,48,49,50,52,52,54,54,55,56,57,57,58,59,59,60,60,59,58,59,59,59,57,52,14,13,14,7,7,7,]

    # # 吉庆街 PT2 Trimble
    # satNO=['G10','G12','G15','G18','G23','G24','G25','G32','R02','R03','R13','R14','R17','R18','E02','E03','E15','E25','J01','J02','J07','C01','C02','C03','C04','C05','C08','C09','C13','C19','C22','C26','C36','C38','C45','C59','C60',]
    # sataz=[5.695719,2.141150,1.234572,3.466468,0.655284,0.784517,2.734763,4.932237,1.146323,6.159398,2.011055,3.117019,3.617197,4.420087,2.643027,0.487225,4.005852,1.574538,1.285161,2.253621,2.722753,2.291231,4.019289,3.269725,2.040623,4.405881,0.000000,4.795414,3.771125,1.823696,1.227903,3.458475,0.948593,3.068705,2.504424,2.373571,4.046478,]
    # satel=[49.422140,24.568506,20.681138,38.133752,78.114005,52.487264,18.008948,27.986491,34.436137,45.594616,59.356550,22.578880,7.524712,25.821396,29.688252,55.828938,39.849848,37.020266,59.250321,40.863837,51.796871,41.371379,42.482420,54.504975,28.821276,22.082302,0.000000,57.534216,68.714669,10.016147,54.112750,20.462324,38.287908,48.969761,62.688745,45.413031,37.406714,]
    # el = [23,23,7,7,7,7,7,7,7,8,8,7,8,14,15,16,41,43,45,46,48,50,50,51,51,51,15,7,7,7,7,7,7,7,7,9,17,18,18,18,17,12,12,12,11,11,10,9,9,9,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,9,9,9,9,9,8,8,7,7,7,7,7,7,7,7,7,7,11,7,7,7,7,7,11,11,11,12,12,12,8,7,7,13,15,17,17,16,16,17,17,18,17,17,17,39,42,42,46,46,46,46,46,46,46,46,45,45,13,13,65,66,67,68,69,69,69,69,69,69,68,68,68,68,68,68,68,67,67,66,66,66,65,65,64,63,63,62,61,60,59,58,57,55,54,52,48,41,]

    # # 吉庆街 PT3 Trimble
    # satNO=['G10','G23','G25','G31','R04','R14','E03','E08','E13','E25','J01','J02','C01','C02','C03','C04','C08','C09','C16','C21','C22','C26','C36','C38','C39','C40','C45','C60',]
    # sataz=[5.725692,2.310284,2.464153,3.969938,5.708624,3.107515,0.976507,5.644246,4.770739,1.149401,1.502024,2.157460,2.292076,4.016333,3.269271,2.041746,0.000000,5.123301,5.873185,6.196994,1.769622,3.550390,0.745166,3.209576,6.218649,2.802871,1.468096,4.040767,]
    # satel=[72.599045,67.798519,36.801498,18.146763,28.567294,50.624394,50.037510,49.425819,28.399695,38.067783,59.567439,50.189144,41.315646,42.306898,54.137841,28.773898,0.000000,56.076299,64.018607,65.981675,44.230318,41.194606,22.797858,37.353139,62.749600,11.649110,72.113088,37.132405,]
    # el = [75,75,75,75,75,75,75,75,74,74,74,74,74,74,73,73,72,72,72,71,70,70,69,68,68,67,66,65,64,63,61,59,57,8,9,14,33,35,37,38,39,41,41,42,43,47,50,50,50,50,50,50,50,50,50,50,49,49,48,48,46,13,7,7,7,7,52,54,54,55,55,56,57,57,58,58,59,59,59,60,60,60,61,61,61,61,61,61,61,61,61,61,61,61,61,60,60,60,59,59,59,58,58,57,57,56,55,55,54,54,52,37,36,34,33,22,23,24,24,25,25,26,26,26,25,24,22,19,17,17,7,7,7,7,35,46,49,49,48,48,48,50,51,51,50,50,57,59,61,63,64,65,66,67,68,68,70,70,70,71,72,72,72,73,73,74,74,74,74,74,74,75,75,75,75,75,75,75,75,75,]

    # # 吉庆街 PT4 Trimble
    # satNO=['G10','G12','G23','G25','G31','G32','R03','R04','R14','E02','E03','E08','E30','J01','J02','J07','C08','C13','C16','C21','C22','C26','C39','C42','C45','C59','C60',]
    # sataz=[3.485359,1.241816,2.646124,2.053260,4.226688,5.724789,1.335392,6.034378,3.076239,2.024810,1.371362,6.013478,3.226161,1.733399,2.039908,2.722438,0.000000,3.739696,6.086413,0.843245,2.114130,3.759886,0.128506,5.580391,0.815631,2.370608,4.036892,]
    # satel=[82.752080,37.866741,47.167000,51.665761,34.399997,50.677135,54.083565,45.137899,77.654545,61.220255,43.952044,58.103805,31.498944,59.383926,57.731772,51.797901,0.000000,42.867735,61.823210,77.784086,31.925708,60.132218,60.991979,35.414167,59.453544,45.530940,36.950514,]
    # el = [13,13,18,22,24,31,33,39,42,44,46,48,50,52,53,54,55,57,58,57,57,56,55,54,53,52,50,49,47,46,44,42,40,37,35,32,26,27,28,30,31,31,33,37,37,37,37,37,37,40,43,44,44,44,44,44,44,44,43,42,42,42,41,40,39,39,37,37,35,35,33,32,31,30,26,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,14,33,46,54,59,64,66,66,66,67,67,68,68,68,68,68,68,68,69,69,69,69,69,68,67,66,65,64,63,61,60,58,56,54,51,48,45,33,33,33,30,19,7,7,7,7,7,7,46,50,52,55,57,59,62,63,65,66,68,68,70,70,71,72,72,73,74,74,75,75,76,75,74,73,72,70,69,68,68,68,69,69,69,69,69,59,24,13,13,7,]

    # WUCE Trimble
    # satNO=['G10','G15','G18','G23','G24','R02','R03','R17','R18','E03','E13','E15','J01','J07','C01','C02','C05','C06','C09','C13','C14','C16','C24','C33','C38','C42','C59','C60',]
    # sataz=[5.482656,0.848594,3.915738,5.908914,1.684280,0.444042,5.500068,3.949323,5.082257,6.072307,5.527005,4.554898,1.099697,2.727289,2.289961,4.026008,4.410271,4.546031,4.309962,4.211822,0.064996,4.937133,2.494258,5.578215,2.730443,0.880165,2.373057,4.059700,]
    # satel=[26.435299,38.149576,64.013893,56.728072,73.133092,42.550759,33.724380,34.745814,37.276324,61.795512,21.783930,55.789314,58.139654,51.897734,41.574071,42.715073,22.224259,66.892344,57.527917,84.577455,59.297068,74.057073,46.607594,45.420113,63.347632,52.054840,45.293848,37.892542,]
    # el = [7,7,7,7,7,7,21,22,23,24,24,25,25,26,26,28,29,30,31,32,33,34,35,36,37,38,39,39,40,41,41,41,42,42,52,54,56,58,59,61,62,63,64,65,66,66,66,65,65,65,65,65,65,65,65,64,64,63,63,63,35,37,38,39,41,42,43,44,45,46,44,44,45,46,46,47,47,48,48,49,49,50,50,50,50,50,51,51,51,51,51,51,51,60,61,61,61,62,62,62,63,63,63,63,63,63,63,62,61,61,60,59,59,57,57,22,22,24,24,23,22,24,22,10,9,9,9,28,29,30,30,31,31,30,53,53,53,53,53,53,52,52,52,52,52,56,58,60,62,63,65,66,66,68,68,69,70,70,71,72,72,72,73,73,74,74,74,73,72,72,72,72,72,72,72,72,70,15,7,7,]
    # el =[7,7,7,7,7,7,21,22,23,24,24,25,25,26,26,28,29,30,31,32,33,34,35,36,37,38,39,39,40,41,41,41,42,42,52,54,56,58,59,61,62,63,64,65,66,66,66,65,65,65,65,65,65,65,65,64,64,63,63,63,35,37,38,39,41,42,43,44,45,46,44,44,45,46,46,47,47,48,48,49,49,50,50,50,50,50,51,51,51,51,51,51,51,60,61,61,61,62,62,62,63,63,63,63,63,63,63,62,61,61,60,59,59,57,57,22,22,24,24,23,22,24,22,10,9,9,9,28,29,30,30,31,31,30,53,53,53,53,53,53,52,52,52,52,52,56,58,60,62,63,65,66,66,68,68,69,70,70,71,72,72,72,73,73,74,74,74,73,72,72,72,72,72,72,72,72,70,15,7,7,]

    # wuce trimble 20220605
    # satNO=['G03','G06','G17','G19','R07','R13','R23','E15','E27','E34','J03','J04','J07','C01','C10','C20','C26','C29','C35','C38','C40','C59',]
    # sataz=[0.710647,5.449455,1.088897,0.157217,1.510958,5.092955,5.413357,5.305881,5.224408,0.572430,1.122987,2.149497,2.724945,2.280195,4.517413,1.895535,5.278136,6.277744,5.251849,5.233196,2.444401,2.394089,]
    # satel=[19.753958,58.903491,60.683524,63.093049,41.925280,34.516130,29.672236,51.719452,40.410157,46.698057,64.601650,54.812188,51.809460,42.279996,78.633499,50.900614,13.770207,59.240633,25.553490,76.382768,68.330537,44.204948,]
    # el = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,41,41,42,43,43,44,44,44,45,45,45,45,45,45,45,44,44,44,44,43,42,41,41,39,38,37,37,37,37,37,37,36,36,35,35,35,34,33,33,32,32,42,42,42,42,42,42,42,42,42,42,42,41,39,37,29,29,30,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,84,84,84,84,84,84,83,83,83,83,83,83,83,82,82,81,81,81,80,79,79,78,77,76,74,73,70,68,64,20,20,29,30,30,30,30,28,21,22,21,17,9,11,11,11,11,11,7,7,7,7,7,7,7,7,7,7,16,18,17,16,7,7,7,7,7,7,7,7,20,21,21,20,17,17,17,17,7,7,]

    # test 2
    # satNO=['G06','G11','G17','G19','R07','R13','R23','E15','E27','E34','J03','J04','C01','C04','C10','C20','C26','C29','C35','C38','C40','C59',]
    # sataz=[5.473816,4.629416,1.113962,0.182522,1.491302,5.075556,5.428147,5.321710,5.212824,0.583120,1.127557,2.146051,2.280131,2.016736,4.493425,1.878034,5.268192,0.014674,5.260673,5.253222,2.460937,2.394024,]
    # satel=[58.975610,29.965375,60.537710,63.416138,42.121169,34.534230,29.958606,51.842530,40.630846,46.385333,64.605784,55.018727,42.282829,30.022927,78.476930,51.194489,13.852014,59.401534,25.917620,76.199844,68.130508,44.208132,]
    # el = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,37,38,39,40,40,41,41,41,41,41,41,41,41,41,41,41,40,39,39,38,37,36,35,34,33,33,33,33,33,33,32,32,31,31,31,30,30,29,41,41,41,41,41,41,83,83,83,83,83,84,84,84,84,84,84,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,85,84,84,84,84,84,84,83,83,83,83,83,83,83,82,82,81,81,81,80,80,79,79,78,77,76,74,72,70,67,63,22,22,30,30,30,31,31,29,22,22,22,17,9,11,11,11,11,11,7,7,7,7,7,7,7,7,7,7,16,18,18,17,16,7,7,7,7,7,7,7,7,21,22,21,20,17,17,17,7,7,]

    # # 2022-06-05 13:48:00
    # satNO=['G02','G11','G19','J03','J04','J07','C07','C08','C10','C13','C29',]
    # sataz=[4.757722,4.652495,0.233123,1.136587,2.139261,2.724935,3.177126,0.000000,4.448668,4.436370,0.054194,]
    # satel=[26.559216,30.729157,64.022744,64.615525,55.418321,51.810084,74.846650,0.000000,78.148768,56.680446,59.705107,]
    # el = [7,7,7,20,22,24,24,25,26,26,26,28,30,32,33,35,37,38,39,41,42,42,44,44,46,46,47,48,48,49,50,50,56,59,61,63,64,65,66,68,68,69,70,71,71,71,71,71,71,71,71,71,70,70,70,70,70,70,69,69,68,68,68,67,41,42,43,44,45,46,47,48,46,46,46,47,47,48,48,49,49,50,50,50,50,50,51,51,51,51,51,51,51,51,51,50,60,61,61,61,61,61,61,61,61,61,61,61,61,61,60,59,59,57,57,55,22,23,24,23,22,23,22,20,9,9,9,10,28,29,30,30,30,29,44,44,43,43,43,43,43,42,42,42,54,55,58,60,61,63,65,65,66,67,68,69,70,70,71,72,72,71,70,70,70,70,70,71,71,72,70,68,63,18,15,15,15,7,7,7,]

    # # 2022-06-05 13:48:01
    # satNO=['G19','J03','J07','C07','C08','C10','C13','C29',]
    # sataz=[4.757722,4.652495,0.233123,1.136587,2.139261,2.724935,3.177126,0.000000,4.448668,4.436370,0.054194,]
    # satel=[26.559216,30.729157,64.022744,64.615525,55.418321,51.810084,74.846650,0.000000,78.148768,56.680446,59.705107,]
    # el = [7,7,7,20,22,24,24,25,26,26,26,28,30,32,33,35,37,38,39,41,42,42,44,44,46,46,47,48,48,49,50,50,56,59,61,63,64,65,66,68,68,69,70,71,71,71,71,71,71,71,71,71,70,70,70,70,70,70,69,69,68,68,68,67,41,42,43,44,45,46,47,48,46,46,46,47,47,48,48,49,49,50,50,50,50,50,51,51,51,51,51,51,51,51,51,50,60,61,61,61,61,61,61,61,61,61,61,61,61,61,60,59,59,57,57,55,22,23,24,23,22,23,22,20,9,9,9,10,28,29,30,30,30,29,44,44,43,43,43,43,43,42,42,42,54,55,58,60,61,63,65,65,66,67,68,69,70,70,71,72,72,71,70,70,70,70,70,71,71,72,70,68,63,18,15,15,15,7,7,7,]

    # 2022-7-5
    # satNO=['G02','G09','G11','G12','G17','G19','R02','E30','J03','J04','C07','C10','C13','C19','C35','C38','C44',]
    # sataz=[5.027704,1.694741,4.963492,5.430576,1.690353,0.941028,0.981736,4.583660,1.202082,2.104687,3.399319,4.039242,4.703094,1.186292,0.868548,5.482853,5.541630,]
    # satel=[34.499590,25.625432,39.986043,26.380419,52.596599,68.001938,39.836368,47.923885,66.888568,60.051270,69.416064,73.151691,57.085023,47.719649,58.975827,71.353988,40.464126,]
    # el = [77,77,77,77,77,77,76,76,76,76,76,76,76,75,75,75,74,74,74,73,41,41,41,42,42,42,42,41,40,38,36,35,32,30,28,26,23,20,17,15,15,13,7,7,7,7,7,7,7,7,7,7,7,7,8,9,45,48,50,52,54,55,56,57,59,60,61,62,63,63,64,65,65,66,66,66,67,68,68,68,68,68,69,69,69,69,69,70,70,70,70,70,70,70,69,69,69,69,69,68,66,65,65,65,65,64,63,63,63,62,61,61,59,59,58,57,55,56,57,57,57,56,55,55,54,54,53,50,46,7,7,7,24,24,24,24,24,22,15,15,15,22,22,22,22,61,63,65,66,67,68,70,70,71,72,71,72,72,73,73,74,74,74,75,75,76,76,76,76,76,76,76,76,77,77,77,77,77,77,77,]

    # satNO=['G02','G04','G06','G09','G11','G17','G19','E15','E30','E34','J03','J04','C07','C10','C13','C24','C29','C38','C40','C44',]
    # sataz=[5.029481,1.124595,6.035338,1.692912,4.965524,1.692994,0.946198,4.959239,4.581476,6.022960,1.202962,2.104083,3.399718,4.037841,4.704804,4.818344,1.672080,5.484724,2.799058,5.542951,]
    # satel=[34.548671,14.412063,58.275731,25.660355,40.035340,52.540064,68.000946,25.963998,47.894616,56.867674,66.890835,60.078275,69.364923,73.108265,57.080427,17.666341,19.321668,71.327484,64.603489,40.525188,]
    # el = [18,22,33,40,66,70,72,75,76,76,76,76,75,75,75,74,74,74,74,73,72,72,71,70,70,69,68,67,66,63,61,60,58,56,54,50,47,16,15,13,10,10,7,7,7,7,7,7,7,7,7,7,35,39,42,46,48,50,52,54,56,57,59,60,61,63,63,64,63,63,63,64,65,65,66,66,66,66,67,67,68,68,68,68,68,68,68,69,69,69,69,69,69,69,69,68,68,68,68,68,68,68,67,67,66,66,66,66,67,67,67,67,67,67,66,66,66,66,66,65,65,65,64,63,61,59,55,9,9,9,9,9,27,28,28,28,28,27,17,17,17,17,25,25,25,25,18,18,20,21,22,44,46,47,48,49,50,51,52,53,54,54,55,55,54,54,55,55,54,51,17,17,17,16,13,13,13,7,16,17,]


    # satNO=['G06','G11','G19','E30','E34','J03','C07','C10','C13','C35','C38','C40',]
    # sataz=[6.038131,4.967240,0.950581,4.579626,6.025071,1.203717,3.400060,4.036657,4.706243,0.874280,5.486288,2.800493,]
    # satel=[58.264601,40.076936,68.000119,47.869488,56.866510,66.893003,69.321332,73.071174,57.076350,58.949203,71.305140,64.567096,]
    # el = [7,7,7,23,24,26,26,27,27,28,30,31,33,35,37,39,41,42,43,44,46,46,48,48,50,50,51,52,52,53,54,54,54,58,60,62,63,65,66,67,68,69,70,70,71,72,72,73,73,73,73,73,73,72,72,72,72,72,72,71,71,70,70,70,69,68,68,67,44,45,46,46,48,46,45,45,46,46,47,48,48,48,48,49,49,49,49,50,50,50,50,50,50,50,50,49,49,59,59,59,60,60,60,60,60,61,61,61,60,59,59,58,57,56,55,54,22,22,23,22,22,23,21,9,9,9,9,28,28,29,29,30,41,41,41,41,41,41,41,41,40,40,40,54,56,59,61,63,64,65,66,68,69,70,70,71,72,72,71,70,70,70,71,71,72,72,72,71,70,68,64,18,18,17,15,15,15,7,7,7,]

    # satNO=['G02','G06','G11','G12','G17','G19','R18','E15','E27','E34','J03','J04','J07','C03','C12','C22','C24','C35','C40',]
    # sataz=[5.032344,6.040668,4.968799,5.427164,1.697270,0.954572,6.024748,4.961568,5.522043,6.026988,1.204411,2.103131,2.726654,3.273398,5.308070,2.809392,4.815836,0.876673,2.801796,]
    # satel=[34.627804,58.254557,40.114723,26.549216,52.448630,67.999206,41.924894,26.020816,16.180616,56.865578,66.895030,60.121744,51.817427,55.105328,19.902665,51.157500,17.626924,58.938234,64.533807,]
    # el = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,46,48,49,50,50,51,52,52,52,53,53,53,54,54,54,54,54,53,53,53,52,52,52,51,50,49,48,47,46,44,44,43,43,43,42,42,42,41,41,41,40,39,39,38,37,37,36,44,44,44,44,44,44,44,43,43,43,42,42,39,37,29,30,30,30,30,30,31,31,31,31,31,31,44,76,78,78,79,80,81,82,83,83,83,83,82,82,82,82,81,81,81,81,81,80,80,79,79,79,78,78,76,76,74,74,72,70,68,65,62,57,20,20,28,29,29,30,29,18,21,21,20,9,10,11,11,11,11,7,7,7,7,7,7,7,7,7,7,15,17,17,17,16,7,7,7,7,7,7,7,7,21,21,21,20,17,17,17,17,7,7,]

    # satNO=['G02','G06','G09','G11','G12','G17','G19','R18','E02','E15','E27','E30','E34','E36','J03','C10','C12','C19','C35','C38',]
    # sataz=[5.033713,6.043219,1.688550,4.970366,5.426158,1.699282,0.958551,6.026710,3.369311,4.962679,5.521301,4.576266,6.028920,0.983228,1.205087,4.034506,5.308963,1.178666,0.879058,5.489147,]
    # satel=[34.665798,58.243992,25.742867,40.152863,26.598789,52.404473,67.997229,41.952380,32.044808,26.048230,16.213980,47.823911,56.864202,31.124477,66.896387,73.004161,19.945124,47.607235,58.926386,71.264697,]
    # el = [17,17,17,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,10,10,9,7,18,18,18,18,18,17,17,17,16,15,15,15,58,62,66,68,70,72,74,75,76,77,78,78,79,79,79,80,80,81,80,80,79,79,79,78,78,77,76,75,74,73,71,70,47,48,48,48,48,48,48,47,44,11,12,12,20,24,27,26,26,26,26,42,45,47,49,50,52,54,55,56,57,58,59,60,61,61,62,63,61,60,59,57,55,53,50,48,44,41,37,33,30,30,39,39,40,40,41,41,40,36,26,26,26,20,13,13,13,15,16,16,16,16,15,15,15,7,7,7,7,7,7,7,7,18,20,19,18,18,7,7,7,7,7,7,7,7,7,22,22,22,20,]

    # huawei p40 无外置天线
    # satNO=['G02','G06','G09','G11','G12','G17','G19','E05','E09','E15','E30','E36','J03','C04','C07','C08','C10','C12','C13','C19','C22','C24','C26','C29','C35','C38','C40',]
    # sataz=[5.027563,6.031767,1.694886,4.963331,5.430672,1.690160,0.940641,2.071543,1.099968,4.957679,4.583822,0.978534,1.202026,2.016204,3.399290,0.000000,4.039339,5.304943,4.702955,1.186474,2.811807,4.820013,3.994745,1.670558,0.868316,5.482693,2.797248,]
    # satel=[34.495778,58.290390,25.622752,39.982193,26.375536,52.600982,68.002283,32.078881,21.006994,25.925962,47.925957,31.263835,66.888593,30.077128,69.419648,0.000000,73.154740,19.754887,57.085258,47.722369,50.898502,17.692181,17.397995,19.374119,58.977227,71.356177,64.649198,]
    # el = [77,76,76,76,76,76,76,76,76,47,49,50,50,51,51,51,51,51,50,50,49,48,46,46,44,43,42,41,39,37,35,33,31,30,27,24,22,19,17,16,16,7,7,7,7,7,7,7,7,7,7,7,7,7,8,9,10,48,50,52,54,55,56,57,59,60,61,62,63,63,64,65,65,66,66,66,67,68,68,68,68,68,69,69,69,69,69,70,70,70,70,70,70,70,69,69,69,69,69,68,68,68,68,68,67,66,63,63,63,62,61,60,59,59,57,57,55,54,54,55,55,55,54,54,53,52,52,50,46,7,7,7,23,24,24,24,24,21,15,15,15,22,22,55,58,60,63,64,66,66,66,67,68,69,70,70,71,72,72,73,74,74,74,74,75,75,75,76,76,76,76,76,76,76,76,76,76,76,77,77,]

    # satNO=['G02','G04','G06','G09','G11','G12','G17','G19','G20','G28','E05','E09','E15','E30','E36','J03','C07','C08','C10','C12','C13','C19','C20','C22','C24','C26','C29','C35','C38','C40',]
    # sataz=[5.028929,1.125105,6.034316,1.693479,4.964893,5.429674,1.692191,0.944619,3.869975,0.000000,2.070510,1.098880,4.958790,4.582141,0.979576,1.202703,3.399593,0.000000,4.038258,5.305837,4.704269,1.184732,0.638278,2.811115,4.818819,3.993817,1.671647,0.870703,5.484135,2.798544,]
    # satel=[34.533603,14.413058,58.280154,25.649566,40.020191,26.425278,52.557435,68.001506,35.600613,0.000000,32.122239,21.003132,25.953157,47.903549,31.232776,66.890263,69.380336,0.000000,73.121407,19.797115,57.081803,47.696992,7.231732,50.972563,17.673736,17.352812,19.336606,58.965990,71.335839,64.616334,]
    # el = [79,79,79,79,79,79,78,78,78,78,78,78,78,78,78,77,77,76,76,76,76,75,74,74,72,71,70,69,68,67,66,65,63,61,58,55,19,19,19,17,13,11,7,7,7,7,7,7,7,7,7,7,7,37,40,42,45,48,50,52,54,55,57,58,59,60,61,62,63,63,64,65,65,66,66,65,65,65,65,65,66,66,66,66,66,66,66,67,67,67,67,67,67,67,66,66,66,66,66,66,66,65,65,65,65,64,63,63,63,62,63,65,65,64,64,64,63,63,63,63,62,61,61,61,59,56,53,9,9,9,8,26,26,26,27,27,28,25,17,17,17,17,24,24,24,24,18,20,21,42,44,45,47,48,50,50,52,52,54,54,53,53,54,54,53,51,48,7,78,78,78,78,78,78,79,79,79,79,79,79,]

    # satNO=['G02','G04','G06','G09','G11','G12','G17','G19','G20','E02','E05','E09','E15','E30','E36','J03','C01','C07','C08','C10','C12','C13','C19','C20','C22','C24','C26','C29','C35','C38','C40',]
    # sataz=[5.029611,1.124471,6.035588,1.692777,4.965673,5.429173,1.693209,0.946612,3.870389,3.370255,2.069995,1.098337,4.959345,4.581299,0.980099,1.203046,2.275669,3.399745,0.000000,4.037715,5.306283,4.704923,1.183865,0.638067,2.810770,4.818221,3.993353,1.672193,0.871899,5.484850,2.799194,]
    # satel=[34.552514,14.411998,58.275128,25.662972,40.039174,26.450170,52.535662,68.001147,35.634877,32.235700,32.143891,21.001259,25.966748,47.892243,31.217357,66.891183,42.310517,69.360552,0.000000,73.104620,19.818274,57.080026,47.684338,7.206636,51.009493,17.664465,17.330127,19.317877,58.960465,71.325724,64.599793,]
    # el = [18,18,22,29,35,39,64,67,69,71,73,74,75,75,75,74,74,74,74,73,72,72,72,71,70,69,68,68,66,65,62,60,58,56,54,51,48,15,15,13,10,10,7,7,7,7,7,7,7,7,7,7,35,39,42,46,48,50,52,54,56,57,59,60,61,63,63,62,62,63,63,64,65,65,66,66,66,66,67,67,68,68,68,68,68,68,68,69,69,69,69,69,69,69,69,68,68,68,68,68,68,68,67,67,66,66,66,68,68,68,68,68,68,67,67,67,66,66,66,66,65,65,65,63,61,59,55,9,9,9,9,9,28,28,28,28,28,27,17,17,17,18,26,26,26,26,26,18,20,21,22,44,46,47,48,49,50,52,52,53,54,55,55,55,56,54,55,55,55,54,50,17,17,16,13,13,13,7,7,17,]

    # huawei p40 外置天线
    # satNO=['G02','G06','G09','G11','G12','G17','G19','E05','E30','J03','C07','C08','C10','C13','C19','C29','C35','C38','C40',]
    # sataz=[5.027568,6.031771,1.694882,4.963337,5.430676,1.690150,0.940628,2.071538,4.583829,1.202013,3.399293,0.000000,4.039354,4.702964,1.186466,1.670555,0.868308,5.482708,2.797243,]
    # satel=[34.495716,58.290036,25.622783,39.982151,26.375351,52.601006,68.002031,32.079037,47.926048,66.888442,69.420005,0.000000,73.154991,57.085299,47.722214,19.374140,58.976966,71.355954,64.649526,]
    # el = [77,76,76,76,76,76,76,76,76,47,49,50,50,51,51,51,51,51,50,50,49,48,46,46,44,43,42,41,39,37,35,33,31,30,27,24,22,19,17,16,16,7,7,7,7,7,7,7,7,7,7,7,7,7,8,9,10,48,50,52,54,55,56,57,59,60,61,62,63,63,64,65,65,66,66,66,67,68,68,68,68,68,69,69,69,69,69,70,70,70,70,70,70,70,69,69,69,69,69,68,68,68,68,68,67,66,63,63,63,62,61,60,59,59,57,57,55,54,54,55,55,55,54,54,53,52,52,50,46,7,7,7,23,24,24,24,24,21,15,15,15,22,22,55,58,60,63,64,66,66,66,67,68,69,70,70,71,72,72,73,74,74,74,74,75,75,75,76,76,76,76,76,76,76,76,76,76,76,77,77,]

    # satNO=['G02','G06','G09','G11','G12','G17','G20','E09','E15','E30','E36','J03','C07','C08','C10','C13','C19','C22','C24','C29','C35','C38','C40',]
    # sataz=[5.028932,6.034318,1.693476,4.964896,5.429676,1.692185,3.869977,1.098878,4.958793,4.582145,0.979574,1.202695,3.399594,0.000000,4.038266,4.704275,1.184728,2.811112,4.818820,1.671645,0.870699,5.484144,2.798540,]
    # satel=[34.533609,58.279981,25.649617,40.020207,26.425217,52.557471,35.600825,21.003071,25.953175,47.903632,31.232690,66.890183,69.380547,0.000000,73.121560,57.081845,47.696928,50.972791,17.673784,19.336649,58.965862,71.335725,64.616529,]
    # el = [79,79,79,79,79,79,78,78,78,78,78,78,78,78,78,77,77,76,76,76,76,75,74,74,72,71,70,69,68,67,66,65,63,61,58,55,19,19,19,17,13,11,7,7,7,7,7,7,7,7,7,7,7,37,40,42,45,48,50,52,54,55,57,58,59,60,61,62,63,63,64,65,65,66,66,65,65,65,65,65,66,66,66,66,66,66,66,67,67,67,67,67,67,67,66,66,66,66,66,66,66,65,65,65,65,64,63,63,63,62,63,65,65,64,64,64,63,63,63,63,62,61,61,61,59,56,53,9,9,9,8,26,26,26,27,27,28,25,17,17,17,17,24,24,24,24,18,20,21,42,44,45,47,48,50,50,52,52,54,54,53,53,54,54,53,51,48,7,78,78,78,78,78,78,79,79,79,79,79,79,]

    # satNO=['G02','G04','G06','G09','G11','G12','G17','G20','E09','E15','E30','E36','J03','C07','C08','C10','C13','C19','C22','C24','C29','C35','C38','C40',]
    # sataz=[5.029615,1.124468,6.035591,1.692772,4.965678,5.429176,1.693200,3.870391,1.098334,4.959348,4.581305,0.980095,1.203034,3.399747,0.000000,4.037728,4.704932,1.183858,2.810766,4.818224,1.672190,0.871892,5.484863,2.799188,]
    # satel=[34.552484,14.411867,58.274828,25.662997,40.039163,26.450032,52.535677,35.635169,21.001115,25.966741,47.892348,31.217176,66.891038,69.360874,0.000000,73.104856,57.080081,47.684195,51.009812,17.664501,19.317893,58.960229,71.325541,64.600081,]
    # el = [18,18,22,29,35,39,64,67,69,71,73,74,75,75,75,74,74,74,74,73,72,72,72,71,70,69,68,68,66,65,62,60,58,56,54,51,48,15,15,13,10,10,7,7,7,7,7,7,7,7,7,7,35,39,42,46,48,50,52,54,56,57,59,60,61,63,63,62,62,63,63,64,65,65,66,66,66,66,67,67,68,68,68,68,68,68,68,69,69,69,69,69,69,69,69,68,68,68,68,68,68,68,67,67,66,66,66,68,68,68,68,68,68,67,67,67,66,66,66,66,65,65,65,63,61,59,55,9,9,9,9,9,28,28,28,28,28,27,17,17,17,18,26,26,26,26,26,18,20,21,22,44,46,47,48,49,50,52,52,53,54,55,55,55,56,54,55,55,55,54,50,17,17,16,13,13,13,7,7,17,]

    # # 2024-1-20 huawei p40  星湖路东门
    # satNO=['G04','G09','G16','G26','G28','G31','E03','E07','E25','E30','J02','C02','C03','C06','C07','C09','C10','C11','C12','C16','C23','C24','C25','C32','C34',]
    # nlos_falg=[0,0,0,1,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,1,0,0,]
    # sataz=[5.272576,5.563997,5.819114,0.550374,1.709528,1.367714,3.032882,5.595135,0.762808,4.481768,1.053510,3.995241,3.270074,0.518849,3.647196,5.963957,3.738281,2.353892,1.231218,0.570871,5.192092,1.262649,0.239104,5.329356,1.852505,]
    # satel=[53.633888,16.883533,69.912205,52.182441,26.278739,46.931450,53.640366,13.525939,33.038566,31.254029,64.156876,41.107012,52.827792,73.959648,59.652525,63.719692,48.673486,33.971985,29.702464,73.272721,36.743768,21.216903,58.937854,21.769713,40.240676,]

    # # 2024-1-20 huawei p40  星湖路西北角
    # satNO=['G03','G04','G16','G25','G26','G28','G29','G31','G32','E02','E03','E05','E08','E24','E25','J02','C02','C03','C06','C09','C13','C14','C16','C24','C25','C26','C33',]
    # nlos_falg=[1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,]
    # sataz=[5.045574,5.527136,3.877154,0.769657,4.397248,0.720017,1.177373,6.186837,2.287958,4.005751,6.156681,2.266193,5.564965,0.737605,0.588070,1.444993,4.011774,3.271876,6.261217,5.697505,3.760077,3.108021,0.023939,0.607651,5.058896,1.059548,3.782302,]
    # satel=[31.382261,12.088600,44.255234,5.611666,74.523607,50.446411,25.156682,58.559792,40.814356,55.727254,81.507453,41.129548,29.952060,15.182910,68.206840,65.068380,41.987598,53.485168,63.040114,54.488311,20.064174,45.018397,62.292001,48.529818,54.967189,5.816095,64.094389,]

    url = '/Users/hzh/Proj/data/smmap-wuhan-wuce1226.db'
    if os.path.exists(url):
        # # 星湖楼东门
        lat = 30.529693923
        lon = 114.3506608794
        # el = SqliteSearch(url,lat,lon)
        # bgurl = "./data/cv/bj3.png"
        bgurl = ""
        # DrawSkyPlot(satNO,sataz,satel,el,bgurl)
        # DrawSkyPlotNLOS(satNO,nlos_falg,sataz,satel,el,bgurl)
        # skytest()

        # # 星湖楼西北角
        # lat = 30.5300684473
        # lon = 114.3497551676
        # el = SqliteSearch(url,lat,lon)
        # bgurl = ""
        # # DrawSkyPlot(satNO,sataz,satel,el,bgurl)
        # DrawSkyPlotNLOS(satNO,nlos_falg,sataz,satel,el,bgurl)
    else:
        print("3dma 数据库不存在")

    # 中北路
    savepath = './data/20210605/polar/pt1.png'
    DrawSkyPlot(satNO,sataz,satel,el,"",savepath,False)
'''
Author: hzh huzihe@whu.edu.cn
Date: 2024-05-03 19:54:57
LastEditTime: 2024-07-06 21:24:42
FilePath: /pyplot/GNSS/NLOS Metric.py
Descripttion: 
'''
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

filename = "./data/202401/log-spp-ublox.res1"

bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.5)
tx, ty = 0.066, 0.668 # 标注信息的位置

class SatInfo(object):
    def __init__(self):
        self.prn = ""
        self.ele, self.snr, self.res_snr, self.resp = 0.0, 0.0, 0.0, 0.0
        self.los = True


def read_res(file):
    sat = []
    with open(file, 'r') as file:
        for line in file:
            if line.startswith('%') or line.startswith('>'):
                continue
            data = line.split(",")

            prn = data[2]
            ele = float(data[11])
            snr = float(data[12])
            res_snr = float(data[13])
            resp = float(data[5])
            los = True if int(data[3]) == 1 else False

            tt = SatInfo()
            tt.prn, tt.ele, tt.snr, tt.res_snr, tt.resp, tt.los = prn, ele, snr, res_snr, resp, los
            sat.append(tt)
    return sat


# 统计每一个高度角区间内nlos和los观测值数量
def ele_statistics(sat):
    # 定义区间和间隔
    intervals = [i for i in range(90)]

    # 初始化计数器字典
    nlos_counts = {interval: 0 for interval in intervals}
    los_counts = {interval: 0 for interval in intervals}

    # 遍历列表中的每个数，统计落在各个区间内的数字个数
    for ss in sat:
        ele = math.floor(ss.ele)
        los = ss.los
        if los:
            los_counts[ele] += 1
        else:
            nlos_counts[ele] += 1

    return los_counts, nlos_counts

# 参考ele修改
def snr_statistics(sat):
    # 定义区间和间隔
    intervals = [i for i in range(60)]

    # 初始化计数器字典
    nlos_counts = {interval: 0 for interval in intervals}
    los_counts = {interval: 0 for interval in intervals}

    # 遍历列表中的每个数，统计落在各个区间内的数字个数
    for ss in sat:
        snr = math.floor(ss.snr)
        los = ss.los
        if los:
            los_counts[snr] += 1
        else:
            nlos_counts[snr] += 1

    return los_counts, nlos_counts

# 参考ele修改
def ressnr_statistics(sat):
    # 定义区间和间隔
    intervals = [i for i in range(-40,20)]

    # 初始化计数器字典
    nlos_counts = {interval: 0 for interval in intervals}
    los_counts = {interval: 0 for interval in intervals}

    # 遍历列表中的每个数，统计落在各个区间内的数字个数
    for ss in sat:
        res_snr = math.floor(ss.res_snr)
        los = ss.los
        if los:
            los_counts[res_snr] += 1
        else:
            nlos_counts[res_snr] += 1

    return los_counts, nlos_counts

# 参考ele修改
def resp_statistics(sat):
    # 定义区间和间隔
    intervals = [i for i in range(-100,100)]

    # 初始化计数器字典
    nlos_counts = {interval: 0 for interval in intervals}
    los_counts = {interval: 0 for interval in intervals}

    # 遍历列表中的每个数，统计落在各个区间内的数字个数
    for ss in sat:
        resp = math.floor(ss.resp)
        resp =math.fabs(resp)
        if resp <= -100 or resp >= 100:
            continue
        los = ss.los
        if los:
            los_counts[resp] += 1
        else:
            nlos_counts[resp] += 1

    return los_counts, nlos_counts
    # return 1,1

# 计算累积概率密度
def ele_cdf(los, nlos):
    # 定义区间和间隔
    intervals = [i for i in range(90)]

    # 初始化计数器字典
    nlos_counts = [0] * 90
    los_counts = [0] * 90

    num_los = sum(los.values())
    num_nlos = sum(nlos.values())

    for i in range(90):
        if i >= 1:
            los_counts[i] = (los_counts[i-1] + los[i])
            nlos_counts[i] = (nlos_counts[i-1] + nlos[i])
        else:
            los_counts[i] = los[i]
            nlos_counts[i] = nlos[i]

    for i in range(90):
        los_counts[i] = los_counts[i] / num_los * 100
        nlos_counts[i] = nlos_counts[i] / num_nlos * 100

    return los_counts, nlos_counts


#todo
def snr_cdf(los, nlos):
    # 定义区间和间隔
    intervals = [i for i in range(60)]

    # 初始化计数器字典
    nlos_counts = [0] * 60
    los_counts = [0] * 60

    num_los = sum(los.values())
    num_nlos = sum(nlos.values())

    for i in range(60):
        if i >= 1:
            los_counts[i] = (los_counts[i-1] + los[i])
            nlos_counts[i] = (nlos_counts[i-1] + nlos[i])
        else:
            los_counts[i] = los[i]
            nlos_counts[i] = nlos[i]

    for i in range(60):
        los_counts[i] = los_counts[i] / num_los * 100
        nlos_counts[i] = nlos_counts[i] / num_nlos * 100

    return los_counts, nlos_counts

#todo
def ressnr_cdf(los, nlos):
    # 定义区间和间隔
    intervals = [i for i in range(-40,20)]

    # 初始化计数器字典
    # nlos_counts = [0] * 70
    # los_counts = [0] * 70
    nlos_counts = {interval: 0 for interval in intervals}
    los_counts = {interval: 0 for interval in intervals}

    num_los = sum(los.values())
    num_nlos = sum(nlos.values())

    for i in range(-40,20):
        if i >= -39:
            los_counts[i] = (los_counts[i-1] + los[i])
            nlos_counts[i] = (nlos_counts[i-1] + nlos[i])
        else:
            los_counts[i] = los[i]
            nlos_counts[i] = nlos[i]

    for i in range(-40,20):
        los_counts[i] = los_counts[i] / num_los * 100
        nlos_counts[i] = nlos_counts[i] / num_nlos * 100

    return los_counts, nlos_counts

#todo
def resp_cdf(los, nlos):
    # 定义区间和间隔
    intervals = [i for i in range(-100,100)]

    # 初始化计数器字典
    nlos_counts = {interval: 0 for interval in intervals}
    los_counts = {interval: 0 for interval in intervals}

    num_los = sum(los.values())
    num_nlos = sum(nlos.values())

    for i in range(-100,100):
        if i >= -99:
            los_counts[i] = (los_counts[i-1] + los[i])
            nlos_counts[i] = (nlos_counts[i-1] + nlos[i])
        else:
            los_counts[i] = los[i]
            nlos_counts[i] = nlos[i]

    for i in range(-100,100):
        los_counts[i] = los_counts[i] / num_los * 100
        nlos_counts[i] = nlos_counts[i] / num_nlos * 100

    return los_counts, nlos_counts

def draw_ele(los, nlos, clos, cnlos):
    plt.figure(dpi=300, figsize=(5.5, 2.3))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    font = {'family': 'Arial', 'size': 10}
    bar_width = 1  # 调整这个值来设置柱的宽度
    top_pannel = 20500

    # 绘图的x轴
    x = [i+0.5 for i in range(90)]


    # 柱状图绘图
    plt.subplot(1, 2, 1)
    plt.bar(x, los.values(), width=bar_width, color='blue', zorder=10, label='LOS')
    plt.bar(x, nlos.values(), width=bar_width, color='red', zorder=12, label='NLOS')

    plt.xlim(0, 90)
    plt.xticks(np.arange(0, 90.1, 10))
    plt.xlabel('Elevation (deg)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    plt.ylim(0, top_pannel)
    plt.yticks(np.arange(0, top_pannel+1, 5000))
    plt.ylabel('Number of observation', fontname='Arial', fontsize=10)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2,2))
    plt.gca().yaxis.set_major_formatter(y_formatter)


    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)
 #   legend.set_bbox_to_anchor((1.05, 1.15))  # 调整图例位置


    # 累积概率密度绘图
    plt.subplot(1, 2, 2)
    plt.plot(clos, color='blue', label='LOS', linestyle='-', linewidth=1.5, zorder=10)
    plt.plot(cnlos, color='red', label='NLOS', linestyle='-', linewidth=1.5, zorder=10)

    plt.xlim(0, 90)
    plt.xticks(np.arange(-0, 90.1, 10))
    plt.xlabel('Elevation (deg)', fontname='Arial', fontsize=10)

    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Cumulative Percentage (%)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)

    # 输出
  #  plt.subplots_adjust(hspace=0.1, wspace=0.3)  # 调整子图间距
    plt.tight_layout()
    plt.savefig('data/202401/cdf/cdf_ele.png', bbox_inches='tight')
    plt.show()

#todo
def draw_snr(los, nlos, clos, cnlos):
    # plt.figure(dpi=300, figsize=(6.2, 4.2))
    plt.figure(dpi=300, figsize=(5.5, 2.3))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    font = {'family': 'Arial', 'size': 10}
    bar_width = 1  # 调整这个值来设置柱的宽度
    top_pannel = 60500

    # 绘图的x轴
    x = [i+0.5 for i in range(60)]


    # 柱状图绘图
    plt.subplot(1, 2, 1)
    plt.bar(x, los.values(), width=bar_width, color='blue', zorder=10, label='LOS')
    plt.bar(x, nlos.values(), width=bar_width, color='red', zorder=12, label='NLOS')

    plt.xlim(0, 60)
    plt.xticks(np.arange(0, 60.1, 10))
    plt.xlabel('C/N0 (dB-Hz)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    plt.ylim(0, top_pannel)
    plt.yticks(np.arange(0, top_pannel+1, 10000))
    plt.ylabel('Number of observation', fontname='Arial', fontsize=10)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2,2))
    plt.gca().yaxis.set_major_formatter(y_formatter)

    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)
 #   legend.set_bbox_to_anchor((1.05, 1.15))  # 调整图例位置


    # 累积概率密度绘图
    plt.subplot(1, 2, 2)
    plt.plot(clos, color='blue', label='LOS', linestyle='-', linewidth=1.5, zorder=10)
    plt.plot(cnlos, color='red', label='NLOS', linestyle='-', linewidth=1.5, zorder=10)

    plt.xlim(0, 60)
    plt.xticks(np.arange(-0, 60.1, 10))
    plt.xlabel('C/N0 (dB-Hz)', fontname='Arial', fontsize=10)

    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Cumulative Percentage (%)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)

    # 输出
  #  plt.subplots_adjust(hspace=0.1, wspace=0.3)  # 调整子图间距
    plt.tight_layout()
    plt.savefig('data/202401/cdf/cdf_snr.png', bbox_inches='tight')
    plt.show()

#todo
def draw_ressnr(los, nlos, clos, cnlos):
    plt.figure(dpi=300, figsize=(5.5, 2.3))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    font = {'family': 'Arial', 'size': 10}
    bar_width = 1  # 调整这个值来设置柱的宽度
    top_pannel = 50500

    # 绘图的x轴
    x = [i+0.5 for i in range(-40,20)]


    # 柱状图绘图
    plt.subplot(1, 2, 1)
    plt.bar(x, los.values(), width=bar_width, color='blue', zorder=10, label='LOS')
    plt.bar(x, nlos.values(), width=bar_width, color='red', zorder=12, label='NLOS')

    plt.xlim(-40, 20)
    plt.xticks(np.arange(-40, 20.1, 10))
    plt.xlabel('∆C/N0 (dB-Hz)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    plt.ylim(0, top_pannel)
    plt.yticks(np.arange(0, top_pannel+1, 10000))
    plt.ylabel('Number of observation', fontname='Arial', fontsize=10)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2,2))
    plt.gca().yaxis.set_major_formatter(y_formatter)


    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)
 #   legend.set_bbox_to_anchor((1.05, 1.15))  # 调整图例位置


    # 累积概率密度绘图
    # 绘图的x轴
    # x = [i+0.5 for i in range(-50,20)]

    plt.subplot(1, 2, 2)
    plt.plot(x, clos, color='blue', label='LOS', linestyle='-', linewidth=1.5, zorder=10)
    plt.plot(x, cnlos, color='red', label='NLOS', linestyle='-', linewidth=1.5, zorder=10)

    plt.xlim(-40, 20)
    plt.xticks(np.arange(-40, 20.1, 10))
    plt.xlabel('∆C/N0 (dB-Hz)', fontname='Arial', fontsize=10)

    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Cumulative Percentage (%)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)

    # 输出
  #  plt.subplots_adjust(hspace=0.1, wspace=0.3)  # 调整子图间距
    plt.tight_layout()
    plt.savefig('data/202401/cdf/cdf_res_snr.png', bbox_inches='tight')
    plt.show()

# todo
def draw_resp(los, nlos, clos, cnlos):
    plt.figure(dpi=300, figsize=(5.5, 2.3))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    font = {'family': 'Arial', 'size': 10}
    bar_width = 1  # 调整这个值来设置柱的宽度
    top_pannel = 40500

    # 绘图的x轴
    x = [i+0.5 for i in range(-100,100)]

    # 柱状图绘图
    plt.subplot(1, 2, 1)
    plt.bar(x, los.values(), width=bar_width, color='blue', zorder=10, label='LOS')
    plt.bar(x, nlos.values(), width=bar_width, color='red', zorder=12, label='NLOS')

    plt.xlim(-100, 100)
    plt.xticks(np.arange(-100, 100.1, 50))
    plt.xlabel('pesudorange residuals (m)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    plt.ylim(0, top_pannel)
    plt.yticks(np.arange(0, top_pannel+1, 10000))
    plt.ylabel('Number of observation', fontname='Arial', fontsize=10)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2,2))
    plt.gca().yaxis.set_major_formatter(y_formatter)


    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)
    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)
 #   legend.set_bbox_to_anchor((1.05, 1.15))  # 调整图例位置


    # 累积概率密度绘图
    plt.subplot(1, 2, 2)
    plt.plot(x, clos, color='blue', label='LOS', linestyle='-', linewidth=1.5, zorder=10)
    plt.plot(x, cnlos, color='red', label='NLOS', linestyle='-', linewidth=1.5, zorder=10)

    plt.xlim(-100, 100)
    plt.xticks(np.arange(-100, 100.1, 40))
    plt.xlabel('pesudorange residuals (m)', fontname='Arial', fontsize=10)

    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Cumulative Percentage (%)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='upper left', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)

    # 输出
  #  plt.subplots_adjust(hspace=0.1, wspace=0.3)  # 调整子图间距
    plt.tight_layout()
    plt.savefig('data/202401/cdf/cdf_resp.png', bbox_inches='tight')
    plt.show()

# todo
def draw_resp_abs(los, nlos, clos, cnlos):
    plt.figure(dpi=300, figsize=(5.5, 2.3))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    font = {'family': 'Arial', 'size': 10}
    bar_width = 1  # 调整这个值来设置柱的宽度
    top_pannel = 40500

    # 绘图的x轴
    x = [i+0.5 for i in range(-100,100)]

    # 柱状图绘图
    plt.subplot(1, 2, 1)
    plt.bar(x, los.values(), width=bar_width, color='blue', zorder=10, label='LOS')
    plt.bar(x, nlos.values(), width=bar_width, color='red', zorder=12, label='NLOS')

    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100.1, 20))
    plt.xlabel('pesudorange residuals (m)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    plt.ylim(0, top_pannel)
    plt.yticks(np.arange(0, top_pannel+1, 10000))
    plt.ylabel('Number of observation', fontname='Arial', fontsize=10)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2,2))
    plt.gca().yaxis.set_major_formatter(y_formatter)


    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)
    plt.legend(loc='upper right', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)
 #   legend.set_bbox_to_anchor((1.05, 1.15))  # 调整图例位置


    # 累积概率密度绘图
    plt.subplot(1, 2, 2)
    plt.plot(x, clos, color='blue', label='LOS', linestyle='-', linewidth=1.5, zorder=10)
    plt.plot(x, cnlos, color='red', label='NLOS', linestyle='-', linewidth=1.5, zorder=10)

    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100.1, 20))
    plt.xlabel('pesudorange residuals (m)', fontname='Arial', fontsize=10)

    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Cumulative Percentage (%)', fontname='Arial', fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), visible=True)  # 隐藏x轴

    ax = plt.gca()
    ax.tick_params(axis='x', direction='in', length=2, which='both', top=True)
    ax.tick_params(axis='y', direction='in', length=2, which='both', right=True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.2)

    plt.legend(loc='lower right', prop=font, ncol=1, labelspacing=0.8, handlelength=2, facecolor='white',
                        columnspacing=0.6, handletextpad=0.25, frameon=True, markerscale=2.0)

    # 输出
  #  plt.subplots_adjust(hspace=0.1, wspace=0.3)  # 调整子图间距
    plt.tight_layout()
    plt.savefig('data/202401/cdf/cdf_resp_abs.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # 读文件
    sat = read_res(filename)

    # 按照区间进行分类
    # los_ele, nlos_ele = ele_statistics(sat)
    # los_snr, nlos_snr = snr_statistics(sat)
    # los_ressnr, nlos_ressnr = ressnr_statistics(sat)
    los_resp, nlos_resp = resp_statistics(sat)

    # 计算累积概率密度
    # clos_ele, cnlos_ele = ele_cdf(los_ele, nlos_ele)
    # clos_snr, cnlos_snr = snr_cdf(los_snr, nlos_snr)
    # clos_ressnr, cnlos_ressnr = ressnr_cdf(los_ressnr, nlos_ressnr)
    clos_resp, cnlos_resp = resp_cdf(los_resp, nlos_resp)


    # 绘图, 绘制哪个把那个注释取消
    # draw_ele(los_ele, nlos_ele, clos_ele, cnlos_ele)
    # draw_snr(los_snr, nlos_snr, clos_snr, cnlos_snr)
    # draw_ressnr(los_ressnr, nlos_ressnr, list(clos_ressnr.values()), list(cnlos_ressnr.values()))
    # draw_resp(los_resp, nlos_resp, list(clos_resp.values()), list(cnlos_resp.values()))
    draw_resp_abs(los_resp, nlos_resp, list(clos_resp.values()), list(cnlos_resp.values()))


    print("处理完成")


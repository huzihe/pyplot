'''
Author: hzh huzihe@whu.edu.cn
Date: 2024-12-01 11:21:32
LastEditTime: 2024-12-02 00:11:14
FilePath: /pyplot/GNSS/statics-bar.py
Descripttion: 
'''
import numpy as np
import matplotlib.pyplot as plt

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 9  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8

def DrawBarGroup(_figname):
    # 数据准备
    labels = ['Trimle Alloy', 'ublox F9P', 'Huawei P40']
    alloy = [0.65, 0.45, 0.50]
    u_blox = [0.55, 0.35, 0.40]
    p40 = [0.60, 0.40, 0.45]

    x = np.arange(len(labels))  # 横坐标位置
    width = 0.15  # 柱状图宽度

    inch = 1/2.54

    plt.figure(dpi=300, figsize=(9*inch, 6*inch))
    plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距        
    ax1 = plt.subplot(1,1,1)
    # 创建图形与子图
    ax1.bar(x - width, alloy, 0.8*width, label='Alloy', color='blue')
    ax1.bar(x, u_blox, 0.8*width, label='u-blox', color='red')
    ax1.bar(x + width, p40, 0.8*width, label='P40', color='LimeGreen')
    ax1.set_ylabel('Silhouette coefficient')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    # 调整布局并显示
    ax1.set_axisbelow(True)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.tight_layout()
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

def DrawBarGroup2(_figname):
    # 数据准备
    labels = ['Trimble Alloy', 'ublox F9P', 'Huawei P40']
    al_SPP = [8.16, 13.68, 16.65]
    al_SM = [7.81, 12.77, 14.69]
    al_SMOP = [6.2, 8.07, 9.51]

    ac_spp = [7.74, 14.98, 14.23]
    ac_sm = [3.25, 4.95, 5.2]
    ac_smop = [2.71, 3.12, 2.89]

    x = np.arange(len(labels))  # 横坐标位置
    width = 0.15  # 柱状图宽度
    inch = 1/2.54
    plt.figure(dpi=300, figsize=(11.9*inch, 8*inch))
    plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距
    
    ax1 = plt.subplot(1,2,1)
    # 创建图形与子图
    ax1.bar(x - width, al_SPP, 0.8*width, color='blue')
    ax1.bar(x, al_SM, 0.8*width, color='red')
    ax1.bar(x + width, al_SMOP, 0.8*width, color='LimeGreen')
    ax1.set_ylabel('Along street RMS (m)')
    plt.ylim(0,20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    # ax1.legend()
    ax1.set_axisbelow(True)
    
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    ax2 = plt.subplot(1,2,2)
    # 右侧子图：Cross Street
    ax2.bar(x - width, ac_spp, 0.8*width, label='SPP', color='blue')
    ax2.bar(x, ac_sm, 0.8*width, label='SM', color='red')
    ax2.bar(x + width, ac_smop, 0.8*width, label='SM-OP', color='LimeGreen')
    ax2.set_ylabel('Cross street RMS (m)')
    plt.ylim(0,20)
    ax2.set_xticks(x)
    ax2.yaxis.set_major_formatter(plt.NullFormatter())  # 隐藏右侧纵坐标数值
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_axisbelow(True)

    # 调整布局并显示
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    plt.gca().tick_params(axis='both', direction='in', length=2, which='both', top=True)

    plt.tight_layout()
    plt.savefig(_figname, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    
    figName = "./data/SM-bar.png"
    # DrawBarGroup(figName)
    figName = "./data/SM-bar2.png"
    DrawBarGroup2(figName)
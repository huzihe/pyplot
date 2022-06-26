# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import xlrd

# commom settings
cm = 1/2.54  # 一厘米等于2.54分之一英寸
halfwidth = 8.8*cm
fullwidth = 17.4*cm
# font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
font = {'size': 8}
plt.figure(figsize=(halfwidth, halfwidth))

filename = './data/spp-sm-pos20220625.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[2]
# 读取第1行标题
title = wind_sheet1.row_values(0)
# 读取第一、三列标题以下的数据 col_values(colx,start_row=0,end_row=none)
x = wind_sheet1.col_values(0, 1, 1200)
y10 = wind_sheet1.col_values(10, 1, 1200)
y11 = wind_sheet1.col_values(11, 1, 1200)
y12 = wind_sheet1.col_values(12, 1, 1200)

y13 = wind_sheet1.col_values(13, 1, 1200)
y14 = wind_sheet1.col_values(14, 1, 1200)
y15 = wind_sheet1.col_values(15, 1, 1200)
# print(x)


# 绘制曲线图
plt.title('Shadowmatching Delt distance', fontsize=12)

#  # 多数据合并到一个图
# plt.ylim((-8,8))
# plt.grid(axis = 'y', color = 'b', linestyle = '--', linewidth = 0.5)
# plt.plot(x, y10, 'g-', label='ref-deltx(m)')
# plt.plot(x,y11,'r-', label='ref-delty(m)')
# plt.plot(x,y12,'b-', label='ref-deltz(m)')
# plt.legend()

# 以下分三个子图绘制
ax1 = plt.subplot(311)    # 分成三行，每行一列，第一行
ax1.set(ylabel='bias (m)')
ax1.set_xticks([])
ax1.set_ylim([-30, 30])
ax1.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
plt.plot(x, y10, 'g-', label='ref-deltx(m)')
plt.plot(x, y13, label='x-delt(m)')
plt.legend(prop=font)      # 设置图例位置

ax2 = plt.subplot(312)
ax2.set(ylabel='bias (m)')
ax2.set_xticks([])
ax2.set_ylim([-30, 30])
ax2.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
plt.plot(x, y11, 'b-', label='ref-delty(m)')
plt.plot(x, y14, label='y-delt(m)')
plt.legend(fontsize=8)

ax3 = plt.subplot(313)
ax3.set(xlabel='Epoch (s)', ylabel='bias (m)')
ax3.set_ylim([-30, 30])
ax3.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
plt.plot(x, y12, 'r-', label='ref-deltz(m)')
plt.plot(x, y15, label='z-delt(m)')
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./data/line-pt1.jpg')   # 保存图片
# plt.show()

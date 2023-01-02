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
plt.figure(figsize=(fullwidth, halfwidth))

filename = './data/0702sm.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[6]
# 读取第1行标题
title = wind_sheet1.row_values(1)
# 读取第一、三列标题以下的数据 col_values(colx,start_row=0,end_row=none)
rows =361
x = wind_sheet1.col_values(0, 2, rows)
y10 = wind_sheet1.col_values(18, 2, rows)
y11 = wind_sheet1.col_values(19, 2, rows)
y12 = wind_sheet1.col_values(17, 2, rows)
y13 = wind_sheet1.col_values(24, 2, rows)
y14 = wind_sheet1.col_values(25, 2, rows)
y15 = wind_sheet1.col_values(23, 2, rows)
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

ylimsetting = [-5, 5]

# 以下分三个子图绘制
ax1 = plt.subplot(311)    # 分成三行，每行一列，第一行
ax1.set(ylabel='x-bias(m)')
ax1.set_xticks([])  # 取消横坐标轴刻度显示
ax1.set_ylim(ylimsetting)
ax1.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
plt.plot(x, y10, 'g-', label='b-delt-sm(m)')
plt.plot(x, y13,'r-', label='b-delt(m)')
plt.legend(prop=font)      # 设置图例位置

ax2 = plt.subplot(312)
ax2.set(ylabel='y-bias (m)')
ax2.set_xticks([])
ax2.set_ylim(ylimsetting)
ax2.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
plt.plot(x, y11, 'g-', label='l-delt-sm(m)')
plt.plot(x, y14,'r-', label='l-delt(m)')
plt.legend(fontsize=8)

ax3 = plt.subplot(313)
ax3.set(xlabel='Epoch (s)', ylabel='z-bias (m)')
ax3.set_ylim(ylimsetting)
ax3.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
plt.plot(x, y12, 'g-', label='h-delt-sm(m)')
plt.plot(x, y15, 'r-', label='h-delt(m)')
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./data/line-trimble-0702-0723-blh.jpg')   # 保存图片
# plt.show()

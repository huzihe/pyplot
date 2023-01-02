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

filename = './data/0710stat.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[0]
# 读取第1行标题
title = wind_sheet1.row_values(0)
# 读取第一、三列标题以下的数据 col_values(colx,start_row=0,end_row=none)
rows = 194
x = wind_sheet1.col_values(0, 1, rows)
y10 = wind_sheet1.col_values(8, 1, rows)
y11 = wind_sheet1.col_values(9, 1, rows)
y12 = wind_sheet1.col_values(10, 1, rows)
# print(x)


# 绘制曲线图
plt.title('GNSS receiver satalite count', fontsize=12)

# ylimsetting = [0, 40]
 # 多数据合并到一个图
plt.ylim((0,40))
plt.grid(axis = 'y', color = 'grey', linestyle = '--', linewidth = 0.5)
plt.plot(x, y10, 'g-', label='alloy')
plt.plot(x,y11,'r-', label='kpl')
plt.plot(x,y12,'b-', label='u-blox')
plt.legend()

# plt.tight_layout()
plt.savefig('./data/line-sat-compare0710.jpg')   # 保存图片
# plt.show()

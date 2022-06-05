# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import xlrd

filename = './data/data analysis.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[1]    
# 读取第1行标题
title = wind_sheet1.row_values(0)
# 读取第一、三列标题以下的数据 col_values(colx,start_row=0,end_row=none)
x = wind_sheet1.col_values(0,1)
y = wind_sheet1.col_values(22,1)
y2 = wind_sheet1.col_values(21,1)
y3 = wind_sheet1.col_values(20,1)
# print(x)
 
# 绘制曲线图
plt.title('Delt distance',fontsize=12)
# plt.plot(x, y, x,y2,x,y3,'g-', label='mean')    # 多数据合并到一个图
plt.subplot(311)    # 分成三行，每行一列，第一行
plt.plot(x,y,'g-', label='fg')
plt.xlim((0,1500))
plt.ylim((0, 180))
plt.legend()      # 设置图例位置

plt.subplot(312)
plt.plot(x,y2,'m-', label='fg-sm')
plt.xlim((0,1500))
plt.ylim((0, 180))
plt.legend()
plt.tight_layout()
# plt.savefig('mean.jpg')   # 保存图片
plt.show()

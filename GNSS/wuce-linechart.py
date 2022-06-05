# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import xlrd

filename = './data/data analysis.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[6]    
# 读取第1行标题
title = wind_sheet1.row_values(0)
# 读取第一、三列标题以下的数据 col_values(colx,start_row=0,end_row=none)
x = wind_sheet1.col_values(0,1)
y8 = wind_sheet1.col_values(8,1)
y9 = wind_sheet1.col_values(9,1)
y10 = wind_sheet1.col_values(10,1)

y13 = wind_sheet1.col_values(13,1)
y14 = wind_sheet1.col_values(14,1)
y15 = wind_sheet1.col_values(15,1)
# print(x)
 
plt.figure(figsize =(17.6,8.8))
# 绘制曲线图
plt.title('Delt distance',fontsize=12)
# plt.plot(x, y, x,y2,x,y3,'g-', label='mean')    # 多数据合并到一个图
plt.subplot(311)    # 分成三行，每行一列，第一行
plt.plot(x,y8,'g-', label='x-ecef(m)')
plt.plot(x,y13, label='spp-x-ecef(m)')
plt.legend()      # 设置图例位置

plt.subplot(312)
plt.plot(x,y9,'m-', label='y-ecef(m)')
plt.plot(x,y14, label='spp-y-ecef(m)')
plt.legend()

plt.subplot(313)
plt.plot(x,y10,'r-', label='z-ecef(m)')
plt.plot(x,y15, label='spp-z-ecef(m)')
plt.legend()

plt.tight_layout()
plt.savefig('./data/test-line.jpg')   # 保存图片
# plt.show()

# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import xlrd
# This is a test

filename = 'data analysis.xlsx'
book_wind = xlrd.open_workbook(filename=filename)
wind_sheet1 = book_wind.sheets()[7]    

eles = wind_sheet1.col_values(2,1)  # 横坐标  高度角
snrs = wind_sheet1.col_values(1,1)  # 纵坐标  SNR值
loss = wind_sheet1.col_values(4,1)
sats = wind_sheet1.col_values(0,1)

for x,y,l,sat in zip(eles,snrs,loss,sats):
    s = sat[0]
    if l == 1:
        if s == 'G':
            plt.plot(x,y,marker='^',color='r')
        elif s == 'R':
            plt.plot(x,y,marker='s',color='r')
        elif s == 'E':
            plt.plot(x,y,marker='d',color='r')
        elif s == 'J':
            plt.plot(x,y,marker='o',color='r')
        elif s == 'C':
            plt.plot(x,y,marker='p',color='r',markersize=4)
        else:
            plt.plot(x,y,'ro')
    if l == 0:
        if s == 'G':
            plt.plot(x,y,marker='^',color='g',markersize=4)
        elif s == 'R':
            plt.plot(x,y,marker='s',color='g',markersize=4)
        elif s == 'E':
            plt.plot(x,y,marker='d',color='g',markersize=4)
        elif s == 'J':
            plt.plot(x,y,marker='o',color='g',markersize=4)
        elif s == 'C':
            plt.plot(x,y,marker='p',color='g',markersize=4)
        else:
            plt.plot(x,y,'go')
# plt.scatter(x,y)
plt.show()

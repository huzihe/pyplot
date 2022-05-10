#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename     :rinexobs.py
@description  :prepare obs for deep learning
@author       :hzh
@time         :2022/05/08 16:42:10
'''

import os
path = os.getcwd()#获取当前路径


# read obs file
with open(path + '/data/IGS000USA_R_20213530105_00M_01S_MO-rtx-sm-mark-0409.rnx', 'r') as f:
    lines_o = f.readlines()

# get obs start line num


def start_num_o():
    for i in range(len(lines_o)):
        if lines_o[i].find('END OF HEADER') != -1:
            start_num = i+1
    return start_num


start_num_o = start_num_o()
# print(start_num_o)

# obs data lines num
o_data_lines_nums = int(len(lines_o)-start_num_o)

# remove obs epoch time records


def o_dic_list():
    o_dic_list = []
    num = 0

    for i in range(o_data_lines_nums):
        sl = lines_o[i+start_num_o]
        flag = sl[0:1]
        if(sl[0:1] != '>'):
            str_line = sl[0:5]
            for j in range(6):
                s_num = int(48*j+5)  # 多频信号开始的位置
                if(sl[s_num:s_num+2] != '  '):
                    str_line += sl[s_num:s_num+44]
                    o_dic_list.append(str_line)
                    break
    return o_dic_list


o_dic_list = o_dic_list()

# print(o_dic_list)


with open(path + '\data\MO-deeplearning.rnx','w',encoding='utf-8') as f_c:
    f_c.write('')

with open(path + '\data\MO-deeplearning.rnx','w',encoding='utf-8') as file:
    for o_dic in o_dic_list:
        file.write(str(o_dic)+'\n')
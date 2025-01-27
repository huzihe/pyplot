#!/usr/bin/python3
from datetime import datetime
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
#plt.rc('font',family='Arial') 

def calMEAN(_numpy_):
    if len(_numpy_) == 0:
        return -1
    mean = 0
    num = 0
    for i in range(len(_numpy_)):
        if not math.isnan(_numpy_[i]):
            num += 1
            mean += _numpy_[i]
    if num == 0:
        return -1
    mean = mean/num
    return mean

def calSTD(_numpy_):
    mean = calMEAN(_numpy_)
    if mean == -1:
        return -1
    std = 0
    num = 0
    for i in range(len(_numpy_)):
        if not math.isnan(_numpy_[i]):
            num += 1
            std += (_numpy_[i]-mean)**2
    if num == 0:
        return -1
    std = math.sqrt(std/num)
    return std

def calRMS(_numpy_):
    if len(_numpy_) == 0:
        return -1
    rms = 0
    num = 0
    for i in range(len(_numpy_)):
        if not math.isnan(_numpy_[i]):
            num += 1
            rms += (_numpy_[i])**2
    if num == 0:
        return -1
    rms = math.sqrt(rms/num)
    return rms    

filename1 = sys.argv[1]
filename2 = sys.argv[2]
pngflname = 'diff_'+filename2[0:15]
#pngflname ='diff'
b = 'RAW-K'
if len(sys.argv) == 4:
  filename3 = sys.argv[3]
  eph3 = len(open(filename3,'r').readlines()) -24
  enu3 = np.full((eph3, 11), np.nan)
  i=0; j=0
  with open(filename3,'r') as f:
      for line in f:
          if line[0] == '#' or line[0:2] == ' %':
              continue          
          if line[21]=='x':
              j=j+1
              enu3[i][6] = 1 #flag
          else:
              enu3[i][6] = 0
          line = line.replace('x', ' ')
          line = line.replace('*', ' ')
          p=line.split()
          enu3[i][0] = int(p[0]); enu3[i][1] = int(p[1]); enu3[i][2] = int(p[2])
          enu3[i][3] = int(p[3]); enu3[i][4] = int(p[4]); enu3[i][5] = float(p[5])
          SIT    = p[6]
          enu3[i][7] = float(p[13]) #E
          enu3[i][8] = float(p[14]) #N
          enu3[i][9] = float(p[15]) #U
          enu3[i][10]= 0
          i=i+1
          if i > eph3-1:
              break

yuzhi = 10
ylm=300
dt='10'
size=22

eph1 = len(open(filename1,'r').readlines()) -24
eph2 = len(open(filename1,'r').readlines()) -24
eph  = max(eph1,eph2)
if len(sys.argv) == 4:
    eph  = max(eph1,eph2,eph3)
enu1 = np.full((eph, 11), np.nan) #year month day hour min sec flag E N U
enu2 = np.full((eph, 11), np.nan)
yuzhi1  = [   yuzhi]*eph
yuzhi2  = [  -yuzhi]*eph
zyuzhi1 = [ 2*yuzhi]*eph
zyuzhi2 = [-2*yuzhi]*eph
i=0; j=0
with open(filename1,'r') as f:
    for line in f:
        if line[0] == '#' or line[0:2] == ' %':
            continue         
        if line[21]=='x':
            j=j+1
            enu1[i][6] = 1 #flag
        else:
            enu1[i][6] = 0
        line = line.replace('x', ' ')
        #line = line.replace('*', ' ')
        if line[21]=='*':
           line = line.replace('*', ' ')
           p=line.split()
           enu1[i][0] = int(p[0]); enu1[i][1] = int(p[1]); enu1[i][2] = int(p[2])
           enu1[i][3] = int(p[3]); enu1[i][4] = int(p[4]); enu1[i][5] = float(p[5])
           SIT    = p[6]
           enu1[i][7]=np.nan
           enu1[i][8]=np.nan
           enu1[i][9]=np.nan
           i=i+1
           continue

        p=line.split()
        enu1[i][0] = int(p[0]); enu1[i][1] = int(p[1]); enu1[i][2] = int(p[2])
        enu1[i][3] = int(p[3]); enu1[i][4] = int(p[4]); enu1[i][5] = float(p[5])
        SIT    = p[6] 
        enu1[i][7] = float(p[13]) #E
        enu1[i][8] = float(p[14]) #N
        enu1[i][9] = float(p[15]) #U
        enu1[i][10]= 0
        i=i+1
        if i > eph-1:
            break
i=0; j=0
with open(filename2,'r') as f:
    for line in f:
        if line[0] == '#' or line[0:2] == ' %':
            continue             
        if line[21]=='x':
            j=j+1
            enu1[i][6] = 1 #flag
        else:
            enu1[i][6] = 0
        line = line.replace('x', ' ')
        #line = line.replace('*', ' ')
        if line[21]=='*':
           line = line.replace('*', ' ')
           p=line.split()
           enu2[i][0] = int(p[0]); enu2[i][1] = int(p[1]); enu2[i][2] = int(p[2])
           enu2[i][3] = int(p[3]); enu2[i][4] = int(p[4]); enu2[i][5] = float(p[5])
           SIT    = p[6]
           enu2[i][7]=np.nan
           enu2[i][8]=np.nan
           enu2[i][9]=np.nan
           i=i+1
           continue

        p=line.split()
        enu2[i][0] = int(p[0]); enu2[i][1] = int(p[1]); enu2[i][2] = int(p[2])
        enu2[i][3] = int(p[3]); enu2[i][4] = int(p[4]); enu2[i][5] = float(p[5])
        SIT    = p[6] 
        enu2[i][7] = float(p[13]) #E
        enu2[i][8] = float(p[14]) #N
        enu2[i][9] = float(p[15]) #U
        enu2[i][10]= 0
        i=i+1
        if i > eph-1:
            break

Time_hms = []
for ieph in range(0, eph):
    Time_hms.append("%04d-%02d-%02d %02d:%02d:%02d" % 
    (enu1[ieph][0], enu1[ieph][1], enu1[ieph][2], enu1[ieph][3], enu1[ieph][4], enu1[ieph][5]))
Time_hms = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in Time_hms]
#myFmt = mdates.DateFormatter('%H:%M:%S')
#myFmt = mdates.DateFormatter('%H:%M')
myFmt = mdates.DateFormatter('%H')

plt.figure(dpi=300,figsize=(14,10))

plt.subplot(3,1,1)
ERMS_1 = calRMS(enu1[:,7])
plt.plot_date(Time_hms,enu1[:,7],fmt='b',label='{}'.format(filename1),linewidth=3)
ERMS_2 = calRMS(enu2[:,7])
plt.plot_date(Time_hms,enu2[:,7],fmt='c',label='{}'.format(filename2),linewidth=3)
if len(sys.argv) == 4:
    ERMS_3 = calRMS(enu3[:,7])
    plt.plot_date(Time_hms,enu3[:,7],fmt='r',label='{}'.format(filename3),linewidth=3)
    print('ERMS : {:>5.1f}  {:>5.1f}  {:>5.1f} [cm]'.format(ERMS_1,ERMS_2,ERMS_3))
else:
    print('ERMS : {:>5.1f}  {:>5.1f} [cm]'.format(ERMS_1,ERMS_2))
plt.ylabel('E [cm]',fontsize=size)
plt.grid(axis="x")
plt.legend(bbox_to_anchor=(1, 1.24),loc='upper right',fontsize=size,ncol=3)
plt.plot_date(Time_hms,enu1[:,10], c='darkgray', fmt='--')
plt.tick_params(labelsize=size)
plt.plot_date(Time_hms,yuzhi1, c='black', fmt='-.')
plt.plot_date(Time_hms,yuzhi2, c='black', fmt='-.')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(pd.date_range(Time_hms[0],Time_hms[len(Time_hms)-1],freq=dt+'T'))
plt.gca().set_xlim(Time_hms[0],Time_hms[-1])
plt.gca().axes.xaxis.set_ticklabels([])
plt.ylim(-1*ylm,ylm)


plt.subplot(3,1,2)
NRMS_1 = calRMS(enu1[:,8])
plt.plot_date(Time_hms,enu1[:,8],fmt='b',label='{}'.format(filename1),linewidth=3)
NRMS_2 = calRMS(enu2[:,8])
plt.plot_date(Time_hms,enu2[:,8],fmt='c',label='{}'.format(filename2),linewidth=3)
if len(sys.argv) == 4:
    NRMS_3 = calRMS(enu3[:,8])
    plt.plot_date(Time_hms,enu3[:,8],fmt='r',label='{}'.format(filename3),linewidth=3)
    print('NRMS : {:>5.1f}  {:>5.1f}  {:>5.1f} [cm]'.format(NRMS_1,NRMS_2,NRMS_3))
else:
    print('NRMS : {:>5.1f}  {:>5.1f} [cm]'.format(NRMS_1,NRMS_2))   
plt.ylabel('N [cm]',fontsize=size)
plt.grid(axis="x")
#plt.legend(loc='upper right',fontsize=size,ncol=3)
plt.plot_date(Time_hms,enu1[:,10], c='darkgray', fmt='--')
plt.tick_params(labelsize=size)
plt.plot_date(Time_hms,yuzhi1, c='black', fmt='-.')
plt.plot_date(Time_hms,yuzhi2, c='black', fmt='-.')
plt.gca().xaxis.set_major_formatter(myFmt)
plt.xticks(pd.date_range(Time_hms[0],Time_hms[len(Time_hms)-1],freq=dt+'T'))
plt.gca().set_xlim(Time_hms[0],Time_hms[-1])
plt.gca().axes.xaxis.set_ticklabels([])
plt.ylim(-1*ylm,ylm)


plt.subplot(3,1,3)
URMS_1 = calRMS(enu1[:,9])
plt.plot_date(Time_hms,enu1[:,9],fmt='b',label='{}'.format(filename1),linewidth=3)
URMS_2 = calRMS(enu2[:,9])
plt.plot_date(Time_hms,enu2[:,9],fmt='c',label='{}'.format(filename2),linewidth=3)
if len(sys.argv) == 4:
    URMS_3 = calRMS(enu3[:,9])
    plt.plot_date(Time_hms,enu3[:,9],fmt='r',label='{}'.format(filename3),linewidth=3)
    print('URMS : {:>5.1f}  {:>5.1f}  {:>5.1f} [cm]'.format(URMS_1,URMS_2,URMS_3))
else:
    print('URMS : {:>5.1f}  {:>5.1f} [cm]'.format(URMS_1,URMS_2))   
plt.ylabel('U [cm]',fontsize=size)
plt.grid(axis="x")
#plt.legend(loc='upper right',fontsize=size,ncol=3)
plt.plot_date(Time_hms,enu1[:,10], c='darkgray', fmt='--')
plt.tick_params(labelsize=size)
plt.plot_date(Time_hms,zyuzhi1, c='black', fmt='-.')
plt.plot_date(Time_hms,zyuzhi2, c='black', fmt='-.')

plt.gca().xaxis.set_major_formatter(myFmt)
plt.gca().set_xlim(Time_hms[0],Time_hms[-1])
plt.xticks(pd.date_range(Time_hms[0],Time_hms[len(Time_hms)-1],freq=dt+'T'))
plt.ylim(-1*ylm,ylm)
plt.xlabel('Time [h]',fontsize=size)
plt.subplots_adjust(left=0.07, bottom=0.05, right=0.95, top=0.95, wspace=0.15, hspace=0.0)
plt.savefig('diff', pad_inches=None, bbox_inches='tight')
#plt.savefig(pngflname, pad_inches=None, bbox_inches='tight')
#plt.show()

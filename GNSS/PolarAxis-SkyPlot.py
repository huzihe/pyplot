import numpy as np
import matplotlib.pyplot as plt

# 数据准备
satNO=['G10','G12','G18','G23','G24','G25','G31','R03','R13','R14','J01','J02','C01','C02','C08','C13','C16','C21','C22','C36',]
sataz=[3.485372,1.241817,2.646128,0.866285,2.053265,4.226688,5.724786,1.335394,6.034376,3.076251,3.545470,1.733403,2.039912,2.292693,5.771697,5.362750,3.739697,6.086410,0.843244,2.114132,]
satel=[82.751940,37.866824,47.166935,14.344508,51.665787,34.399804,50.677101,54.083663,45.137903,77.654447,24.382253,59.384005,57.731816,41.278370,61.157935,54.865179,42.867580,61.823247,77.784226,31.925710,]
el = [13,13,18,22,24,31,33,39,42,44,46,48,50,52,53,54,55,57,58,57,57,56,55,54,53,52,50,49,47,46,44,42,40,37,35,32,26,27,28,30,31,31,33,37,37,37,37,37,37,40,43,44,44,44,44,44,44,44,43,42,42,42,41,40,39,39,37,37,35,35,33,32,31,30,26,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,14,33,46,54,59,64,66,66,66,67,67,68,68,68,68,68,68,68,69,69,69,69,69,68,67,66,65,64,63,61,60,58,56,54,51,48,45,33,33,33,30,19,7,7,7,7,7,7,46,50,52,55,57,59,62,63,65,66,68,68,70,70,71,72,72,73,74,74,75,75,76,75,74,73,72,70,69,68,68,68,69,69,69,69,69,59,24,13,13,7,]
az=[]
index = 0
for item in el:
    az.append((360 / len(el)) * index *np.pi / 180)
    index +=1
az.append(2 * np.pi);
el.append(el[0])

img = plt.imread("bj2.png")
axes_coords = [0.123, 0.111, 0.779, 0.77] # plotting full width and height


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# # 绘制背景图片
# ax_image = fig.add_axes(axes_coords, label="ax image")
# ax_image.imshow(img, alpha=.4)
# ax_image.axis('off')  # don't show the axes ticks/lines/etc. associated with the image

# 天空阴影图极坐标绘制
ax.patch.set_facecolor('0.85')   # 底色设置为灰色
ax.plot(az,el)   # 绘制建筑边界
ax.fill(az,el,'w')   #中间天空填充白色

# 绘制卫星，用不同形状表示不同星座
for x,y,t in zip(sataz,satel,satNO):
    s = t[0]
    if s == 'G':
        ax.plot(x,y,marker='^',color='g',markersize=10)
    elif s == 'R':
        ax.plot(x,y,marker='s',color='m',markersize=10)
    elif s == 'E':
        ax.plot(x,y,marker='d',color='b',markersize=10)
    elif s == 'J':
        ax.plot(x,y,marker='o',color='y',markersize=10)
    elif s == 'C':
        ax.plot(x,y,marker='p',color='r',markersize=10)
    else:
        ax.plot(x,y,'ro')
    ax.text(x, y, t,horizontalalignment='left',verticalalignment='bottom',color='darkslategray',fontsize=11)
# 绘制卫星结束

ax.set_rmax(2)
ax.set_rticks([90, 80, 60, 40, 20])  # Less radial ticks
ax.set_rlabel_position(0)  # Move radial labels away from plotted line
ax.set_theta_zero_location('N')  #0°位置为正北方向
ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
ax.set_theta_direction(-1)   # 顺时针
ax.set_rlim(90,0)
# ax.grid(True)

ax.set_title("A skymask on a polar axis", va='bottom')
plt.show()

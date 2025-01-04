import mxnet as mx
from mxnet import image
import gluoncv
import matplotlib.pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete, plot_image
import matplotlib.image as mpimg
import numpy as np
import math
import os

# using cpu
ctx = mx.cpu()


def get_files(path):
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(filepath, filename))


# get_files('./data/cv')


def skysegmentaion(idir, filename, flag):
    # img = image.imread('./data/cv/cv1.jpg')
    file = os.path.join(idir, filename)
    if not os.path.exists(file):
        return

    # 输出中间过程文件
    if flag:
        imdir = os.path.dirname(idir) + "/im/"
        if not os.path.exists(imdir):
            os.mkdir(imdir)

    odir = os.path.dirname(idir) + "/out/"
    if not os.path.exists(odir):
        os.mkdir(odir)

    img = image.imread(file)
    img = test_transform(img, ctx)

    model2 = gluoncv.model_zoo.get_model("deeplab_resnet101_ade", pretrained=True)
    output = model2.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, "ade20k")
    sky = predict == 2
    print(
        file + " 天空率为:", str(len(predict[sky]) / (predict.shape[0] * predict.shape[1]))
    )

    if flag:
        base = mpimg.imread(file)
        plt.figure(figsize=(10, 5))
        # plt.imshow(base)
        plt.imshow(mask, alpha=0.5)
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(imdir, filename), dpi=250)
        print(file + " segmentation is done!")

    az = []
    el = []
    rows = predict.shape[0]
    cols = predict.shape[1]
    orientation = 180
    for col in range(360):
        frac = math.modf((col + orientation) / 360)
        cole = int(frac[0] * cols)
        # for row in range(int(rows / 2)):
        #     if predict[row, cole] != 2:
        for row in range(int(rows / 2), 1, -2):
            if predict[row, cole] == 2:
                az.append(col * np.pi / 180)
                el.append(90 - row * 2 / rows * 90)
                ele = row * 2 / predict.shape[0] * 90
                break
    az.append(az[0])
    el.append(el[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4.4, 4.4))
    # 天空阴影图极坐标绘制
    ax.patch.set_facecolor("0.85")  # 底色设置为灰色
    ax.plot(az, el)  # 绘制建筑边界
    ax.fill(az, el, "w")  # 中间天空填充白色

    ax.set_rmax(2)
    ax.set_rticks([90, 80, 60, 40, 20])  # Less radial ticks
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_theta_zero_location("N")  # 0°位置为正北方向
    ax.set_thetagrids(np.arange(0.0, 360.0, 30.0))
    ax.set_theta_direction(-1)  # 顺时针
    ax.set_rlim(90, 0)

    plt.savefig(os.path.join(odir, filename))
    print(file + " skyplot is done!")


path = "./data/sky/res"

# filenames = os.walk(path)
# for filename in filenames:
#     skysegmentaion(path,filename,opath)
filenames = os.listdir(path)
for i in filenames:
    skysegmentaion(path, i, True)
print("it's done!")

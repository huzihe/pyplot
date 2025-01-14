'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-05-29 21:59:29
LastEditTime: 2025-01-04 20:44:58
FilePath: /pyplot/PanoCV/gluonCV.py
Descripttion: 
'''
import mxnet as mx
from mxnet import image
import gluoncv
import matplotlib.pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete, plot_image
import matplotlib.image as mpimg
import numpy as np

def deeplab_citys(input,outimg):
    # using cpu
    ctx = mx.cpu()

    img = image.imread(input)
    img = test_transform(img, ctx)

    # model1 = gluoncv.model_zoo.get_model('deeplab_resnet101_citys', pretrained=True)
    # output = model1.predict(img)
    # predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    # sky = (predict==10)
    # print('天空率为:',str(len(predict[sky])/(predict.shape[0]*predict.shape[1])))

    # mask = get_color_pallete(predict, 'citys')
    # base = mpimg.imread('./data/cv/cv2.jpg')
    # plt.figure(figsize=(10,5))
    # plt.imshow(base)
    # plt.imshow(mask,alpha=0.5)
    # plt.axis('off')
    # plt.savefig('./data/cv/cv2_deeplab_citys.jpg',dpi=250,bbox_inches='tight')

    model2 = gluoncv.model_zoo.get_model("deeplab_resnet101_ade", pretrained=True)
    output = model2.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, "ade20k")
    sky = predict == 2
    print("天空率为:", str(len(predict[sky]) / (predict.shape[0] * predict.shape[1])))
    # mask = get_color_pallete(predict[sky], "ade20k")

    base = mpimg.imread(input)
    plt.figure(figsize=(10, 5))
    # plt.imshow(base)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(outimg, dpi=250)
    # print("it's done!")

if __name__ == "__main__":
    inImg = './data/cv/cv3.jpg'
    outImg = './data/cv/cv3_deeplab_ade-t.jpg'

    # inImg = './data/cv/cv2.jpg'
    # outImg = './data/cv/cv2_deeplab_ade-t1.jpg'

    deeplab_citys(inImg,outImg)


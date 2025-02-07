'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-05-29 21:59:29
LastEditTime: 2025-02-07 11:17:08
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
import time
import os

def get_files(path):
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(filepath, filename))

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
    plt.imshow(base)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(outimg, dpi=250)
    # print("it's done!")

def skysegmentation(modelname,input,outimg):
    # using cpu
    ctx = mx.cpu()

    img = image.imread(input)
    img = test_transform(img, ctx)

    # 记录开始时间
    start_time = time.time()

    model2 = gluoncv.model_zoo.get_model(modelname, pretrained=True)
    output = model2.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    mask = get_color_pallete(predict, "ade20k")
    sky = predict == 2
    print("天空率为:", str(len(predict[sky]) / (predict.shape[0] * predict.shape[1])))

    # 记录结束时间
    end_time = time.time()
    print(modelname + " 耗时:", end_time - start_time)

    base = mpimg.imread(input)
    plt.figure(figsize=(10, 5))
    plt.imshow(base)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(outimg, dpi=250)

def skysegmentation2(modelname, input, outimg):
    # using cpu
    ctx = mx.cpu()

    img = image.imread(input)
    img = test_transform(img, ctx)

    # 记录开始时间
    start_time = time.time()

    model2 = gluoncv.model_zoo.get_model(modelname, pretrained=True)
    output = model2.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    
    # unique_labels = np.unique(predict)
    # print("识别结果中的类别编号:", unique_labels)
    # 仅保留天空（2）和树木（4）类别
    sky = 2
    tree = 4
    mask = (predict == sky) | (predict == tree)
    filtered_predict = predict * mask  # 其他类别设为0
    filtered_mask = get_color_pallete(filtered_predict, "ade20k")
    # sky = predict == 2
    # print("天空率为:", str(len(predict[sky]) / (predict.shape[0] * predict.shape[1])))

    # 记录结束时间
    end_time = time.time()
    print(modelname + " 耗时:", end_time - start_time)

    base = mpimg.imread(input)
    plt.figure(figsize=(10, 5))
    plt.imshow(base)
    plt.imshow(filtered_mask, alpha=0.5)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(outimg, dpi=250)

def skysegmentation_onlysky(modelname,input,outimg):
    # using cpu
    ctx = mx.cpu()

    img = image.imread(input)
    img = test_transform(img, ctx)

    model2 = gluoncv.model_zoo.get_model(modelname, pretrained=True)
    output = model2.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    sky_mask = (predict == 2).astype("uint8")  # 将天空区域转换为二值 mask (1 表示天空，0 表示其他)
    base = mpimg.imread(input)
    # 创建一个空白的 RGB 图像用于显示天空 mask
    colored_mask = np.zeros_like(base)  # 初始化一个空白图片（与输入图片相同大小）
    colored_mask[:, :, 2] = sky_mask * 255  # 只给天空区域着色（蓝色通道）

    plt.figure(figsize=(10, 5))
    # plt.imshow(base)
    plt.imshow(colored_mask, alpha=0.7)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(outimg, dpi=250)
    # print("it's done!")

def deeplab_citys_sky_and_trees(modelname,input, outimg):
    # 使用 CPU
    ctx = mx.cpu()

    # 加载图像并预处理
    img = image.imread(input)
    img = test_transform(img, ctx)

    # 加载预训练的 DeepLabv3 模型
    model = gluoncv.model_zoo.get_model(modelname, pretrained=True)

    # 进行预测
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    # 生成天空和树木的掩码
    if modelname in {"deeplab_resnet101_ade", "fcn_resnet50_ade", "deeplab_resnest200_ade"}:
        sky_idx = 2
    elif modelname == "deeplab_resnet50_citys":
        sky_idx = 10
    else:
        sky_idx = None  # 默认值，防止未匹配时报错

    # tree_idx = 4  # 树木的类别索引
    sky_mask = (predict == sky_idx).astype("uint8")  # 天空掩码
    # tree_mask = (predict == tree_idx).astype("uint8")  # 树木掩码

    # 加载原始图像
    base = mpimg.imread(input)

    # 创建一个空白的 RGB 图像用于显示掩码
    colored_mask = np.zeros_like(base)  # 初始化一个空白图像（与输入图像相同大小）

    # 将天空区域标记为蓝色（BGR 格式）
    # colored_mask[:, :, 2] = sky_mask * 255  # 蓝色通道
    colored_mask[sky_mask == 1] = [255, 255, 255]  # 将天空区域标记为白色（BGR 格式）

    # # 将原始图像与彩色掩码叠加
    # result = base * 0.7 + colored_mask * 0.3  # 调整透明度以突出显示目标区域
    result = colored_mask  # 调整透明度以突出显示目标区域
    result = np.clip(result, 0, 255).astype("uint8")  # 确保像素值在 0-255 范围内

    # 保存结果
    mpimg.imsave(outimg, result)

    print(f"结果已保存至: {outimg}")

def deeplab_citys_onlysky(input,outimg):
    # using cpu
    ctx = mx.cpu()

    img = image.imread(input)
    img = test_transform(img, ctx)

    model2 = gluoncv.model_zoo.get_model("deeplab_resnet50_ade", pretrained=True)
    output = model2.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    sky_mask = (predict == 2).astype("uint8")  # 将天空区域转换为二值 mask (1 表示天空，0 表示其他)
    base = mpimg.imread(input)
    # 创建一个空白的 RGB 图像用于显示天空 mask
    colored_mask = np.zeros_like(base)  # 初始化一个空白图片（与输入图片相同大小）
    colored_mask[:, :, 2] = sky_mask * 255  # 只给天空区域着色（蓝色通道）
    
    # # 从 ADE20K 调色板中获取天空类别的颜色
    # mask_image = get_color_pallete(predict, "ade20k")  # 调用 ADE20K 调色板生成的着色图
    # mask_image = np.array(mask_image)  # 转换为 numpy 格式

    # # 提取天空区域的颜色
    # colored_mask = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)  # RGBA 图像
    # colored_mask[:, :, :3] = mask_image[:, :, :3]  # 复制 RGB 颜色
    # colored_mask[:, :, 3] = sky_mask * 128  # 只设置天空区域的透明度（半透明）

    plt.figure(figsize=(10, 5))
    # plt.imshow(base)
    plt.imshow(colored_mask, alpha=0.7)
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(outimg, dpi=250)
    # print("it's done!")



if __name__ == "__main__":
    inImg = './data/cv/sky/res/cv4.jpg'
    outImg = './data/cv/sky/out/cv4_deeplab_ade_sky.jpg'

    # inImg = './data/cv/cv2.jpg'
    # outImg = './data/cv/cv2_deeplab_ade-t1.jpg'

    # deeplab_citys(inImg,outImg)
    # deeplab_citys_onlysky(inImg,outImg)


    # outImg = './data/cv/sky/out/cv4_deeplab_resnet50_citys_sky.jpg'
    # skysegmentation("deeplab_resnet50_citys",inImg,outImg)

    # inImg = './data/cv/sky/res/cv3.jpg'
    # outImg = './data/cv/sky/out/cv3_deeplab_resnet50_citys_sky.jpg'
    # skysegmentation("deeplab_resnet50_citys",inImg,outImg)

    # outImg = './data/cv/sky/out/cv3_deeplab_resnet50_ade_sky.jpg'
    # skysegmentation("deeplab_resnet50_ade",inImg,outImg)

    # outImg = './data/cv/sky/out/cv3_deeplab_v3b_plus_wideresnet_citys_sky.jpg'
    # skysegmentation("deeplab_v3b_plus_wideresnet_citys",inImg,outImg)

    # outImg = './data/cv/sky/out/cv3_fcn_resnet50_ade_sky.jpg'
    # skysegmentation("fcn_resnet50_ade",inImg,outImg)


    # path = "./data/cv/sky/res"
    # outPath = "./data/cv/sky/out_deeplab"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     # skysegmentaion(path, i, True)
    #     skysegmentation("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res1"
    # outPath = "./data/cv/sky/out_deeplab"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     skysegmentation("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_cityscapes"
    # outPath = "./data/cv/sky/out_deeplab"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".png", "_deeplab_resnet101_ade_sky_tree.png")))
    #     # deeplab_citys_onlysky(os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_onlysky.jpg")))
    #     # skysegmentation("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_cityscapes"
    # outPath = "./data/cv/sky/out_fcn"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".png", "_fcn_resnet50_ade_sky_tree.png")))
    #     # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_egova"
    # outPath = "./data/cv/sky/out_deeplab"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     skysegmentation("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_egova"
    # outPath = "./data/cv/sky/out_fcn"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res"
    # outPath = "./data/cv/sky/out_fcn"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     # skysegmentaion(path, i, True)
    #     skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_t.jpg")))
    # print("it's done!")

# # egova
#     path = "./data/cv/sky/res_egova/"
#     outPath = "./data/cv/sky/out_egova/"
#     filenames = os.listdir(path)
#     for i in filenames:
#         # deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
#         # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
#         # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
#         skysegmentation("psp_resnet101_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet101_citys_sky.jpg")))
#         skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky.jpg")))
#         skysegmentation("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky.jpg")))
#     print("it's done!")

# # ade20k
#     path = "./data/cv/sky/res_ade20k/"
#     outPath = "./data/cv/sky/out_ade20k/"
#     filenames = os.listdir(path)
#     for i in filenames:
#         deeplab_citys_sky_and_trees("deeplab_resnest200_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnest200_ade_sky.jpg")))
#         deeplab_citys_sky_and_trees("psp_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet50_ade_sky.jpg")))
#         # deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
#         deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
#         deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
#         # skysegmentation("psp_resnet101_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet101_citys_sky.jpg")))
#         # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky.jpg")))
#         # skysegmentation("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky.jpg")))
#     print("it's done!")

# egova
    path = "./data/cv/sky/res_egova1/"
    outPath = "./data/cv/sky/out_egova1/"
    filenames = os.listdir(path)
    for i in filenames:
        skysegmentation2("deeplab_resnest200_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnest200_ade_sky_tree.jpg")))
        skysegmentation2("psp_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet50_ade_sky_tree.jpg")))
        # skysegmentation("psp_resnet101_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet101_citys_sky.jpg")))
        skysegmentation2("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
        # skysegmentation("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky.jpg")))
        skysegmentation2("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    print("it's done!")

# egova
    path = "./data/cv/sky/res_wuce/"
    outPath = "./data/cv/sky/out_wuce1/"
    filenames = os.listdir(path)
    for i in filenames:
        skysegmentation2("deeplab_resnest200_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnest200_ade_sky_tree.jpg")))
        skysegmentation2("psp_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet50_ade_sky_tree.jpg")))
        # skysegmentation("psp_resnet101_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet101_citys_sky.jpg")))
        skysegmentation2("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
        # skysegmentation("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky.jpg")))
        skysegmentation2("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    print("it's done!")

# # wuce
#     path = "./data/cv/sky/res/"
#     outPath = "./data/cv/sky/out_wuce/"
#     filenames = os.listdir(path)
#     for i in filenames:
#         # skysegmentation("mobilenetv3_large", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_mobilenetv3_large_sky.jpg")))
#         skysegmentation("psp_resnet101_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_psp_resnet101_citys_sky.jpg")))
#         skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky.jpg")))
#         skysegmentation("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky.jpg")))
#         skysegmentation("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
#     print("it's done!")

# skyfinder
    # path = "./data/cv/sky/res_skyfinder/684"
    # outPath = "./data/cv/sky/skyfinder/"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
    #     # # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_skyfinder/3297"
    # outPath = "./data/cv/sky/skyfinder/"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
    #     # # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_skyfinder/4181"
    # outPath = "./data/cv/sky/skyfinder/"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
    #     # # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_skyfinder/9483"
    # outPath = "./data/cv/sky/skyfinder/"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
    #     # # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_skyfinder/11331"
    # outPath = "./data/cv/sky/skyfinder/"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
    #     # # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")

    # path = "./data/cv/sky/res_skyfinder/18590"
    # outPath = "./data/cv/sky/skyfinder/"
    # filenames = os.listdir(path)
    # for i in filenames:
    #     deeplab_citys_sky_and_trees("deeplab_resnet50_citys", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet50_citys_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("deeplab_resnet101_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky_tree.jpg")))
    #     # deeplab_citys_sky_and_trees("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_fcn_resnet50_ade_sky_tree.jpg")))
    #     # # skysegmentation("fcn_resnet50_ade", os.path.join(path, i), os.path.join(outPath, i.replace(".jpg", "_deeplab_resnet101_ade_sky.jpg")))
    # print("it's done!")
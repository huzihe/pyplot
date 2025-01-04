'''
Author: hzh huzihe@whu.edu.cn
Date: 2024-12-31 23:10:38
LastEditTime: 2025-01-01 16:17:35
FilePath: /pyplot/PanoCV/openCV.PY
Descripttion: 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 字体调整
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 10  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
# plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
# plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
# plt.rcParams['legend.fontsize'] = 8

inch = 1/2.54

def Otsu(inPath,outPath):
    # 1. 加载图像并转换为灰度图
    image = cv2.imread(inPath)  # 替换为你的图像路径
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊（可选，但推荐）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 使用 Otsu 阈值分割
    # 返回值 ret 为 Otsu 方法自动计算的最佳阈值
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.adaptiveThreshold

    # 4. 显示结果
    print(f'Otsu 阈值: {ret}')  # 打印最佳阈值
    # plt.figure(dpi=300, figsize=(6*inch, 9*inch))
    # plt.subplots_adjust(wspace =0.1, hspace =0)#调整子图间距 
    plt.figure(figsize=(6, 8))
    plt.subplot(3, 1, 1)
    # plt.title('Original Image')
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(3, 1, 2)
    # plt.title('Gray Image')
    # plt.title('Gray Image')
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('Gray Image')
    plt.imshow(gray, cmap='gray')

    plt.subplot(3, 1, 3)
    # plt.title('Otsu Thresholding')
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('Otsu Result')
    plt.imshow(binary, cmap='gray')


    plt.tight_layout()
    plt.savefig(outPath)
    plt.show()

def OtsuGaussian(inPath,outPath):
    # 1. 加载图像并转换为灰度图
    image = cv2.imread(inPath)  # 替换为你的图像路径
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊（可选）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 自适应阈值分割
    # 参数说明：
    # cv2.ADAPTIVE_THRESH_MEAN_C：使用局部均值作为阈值计算方法
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C：使用加权局部均值（高斯权重）作为阈值计算方法
    # Block Size：定义邻域的大小（需为奇数），如 11
    # C：常量，用于调整阈值结果，值越大结果越暗
    adaptive_mean = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    adaptive_gaussian = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 4. 显示结果
    # plt.figure(figsize=(5,))
    plt.figure(dpi=300, figsize=(8, 10))
    plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距 

    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('Original Image')

    plt.subplot(3, 1, 2)
    # plt.title('Adaptive Mean Thresholding')
    plt.imshow(adaptive_mean, cmap='gray')
    plt.yticks([])
    plt.xticks([])
    # plt.axis('off')
    plt.ylabel('Adaptive Mean Thresholding', )

    plt.subplot(3, 1, 3)
    # plt.title('Adaptive Gaussian Thresholding')
    plt.imshow(adaptive_gaussian, cmap='gray')
    plt.ylabel('Gaussian Thresholding')
    plt.yticks([])
    plt.xticks([])
    # plt.axis('off')

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1)
    plt.savefig(outPath)
    plt.show()

if __name__ == "__main__":
    inImage = './data/cv/cv2.jpg'
    outImage = './data/cv/cv2_OtsuGaussian.jpg'
    # OtsuGaussian(inImage,outImage)

    outOtsu= './data/cv/cv2_Otsu.jpg'
    Otsu(inImage,outOtsu)
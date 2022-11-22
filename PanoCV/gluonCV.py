import mxnet as mx
from mxnet import image
import gluoncv
import matplotlib.pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete,plot_image
import matplotlib.image as mpimg
import numpy as np
# using cpu
ctx = mx.cpu()

img = image.imread('./data/cv/cv1.jpg')
img = test_transform(img,ctx)


model1 = gluoncv.model_zoo.get_model('deeplab_resnet101_citys', pretrained=True)
output = model1.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
sky = (predict==10)
print('天空率为:',str(len(predict[sky])/(predict.shape[0]*predict.shape[1])))


mask = get_color_pallete(predict, 'citys')
base = mpimg.imread('./data/cv/cv1.jpg')
plt.figure(figsize=(10,5))
plt.imshow(base)
plt.imshow(mask,alpha=0.5)
plt.axis('off')
plt.savefig('./data/cv/cv1_deeplab_citys.jpg',dpi=250,bbox_inches='tight')

model2 = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True)
output = model2.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
mask = get_color_pallete(predict, 'ade20k')
sky = (predict==2)
print('天空率为:',str(len(predict[sky])/(predict.shape[0]*predict.shape[1])))

base = mpimg.imread('./data/cv/cv1.jpg')
plt.figure(figsize=(10,5))
plt.imshow(base)
plt.imshow(mask,alpha=0.5)
plt.axis('off')
plt.savefig('./data/cv/cv1_deeplab_ade.jpg',dpi=250,bbox_inches='tight')

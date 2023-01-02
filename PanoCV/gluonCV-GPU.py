from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import mxnet as mx

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True,ctx=mx.gpu(0))

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxes = net(x.as_in_context(mx.gpu(0)))

ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()
plt.savefig('./data/cv/gpu1.jpg',dpi=250,bbox_inches='tight')
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image
from conv import SimpleConvNet


def get_image():
    path_to_image='testdata/7.jpg'
    im = Image.open(path_to_image)
    im = im.convert('L')

# 转换图像到mnist的大小28*28
    im = im.resize((28,28))
# 获取图像长宽
    (width, height) = im.size
# 将图像数据转化位mnist
    data_image=np.ones((1,1,28,28),int)    
    for x in range(0,width):
        for y in range(0,height):
           data_image[0][0][x][y]=255-im.getpixel((y,x))



    return data_image

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


con=SimpleConvNet()
con.load_params()
x=get_image()
y=con.predict(x)
p=np.argmax(y)
print(p)
print(y)

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image

# 接下来打开图片，并且将像转化为8bit黑白像素
def get_image():
    path_to_image='lalala.jpg'
    im = Image.open(path_to_image)
    im = im.convert('L')

# 转换图像到mnist的大小28*28
    im = im.resize((28,28))
# 获取图像长宽
    (width, height) = im.size
# 将图像数据转化位mnist
    data_image=[]
    
    for x in range(0,width):
        for y in range(0,height):
            data_image.append(255 -im.getpixel((y,x)))
    
    return data_image


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("params.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

'''
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
'''
def predict(self, x):
    for layer in self.layers.values():
        x = layer.forward(x)

    return x


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

'''
network=init_network()
x=get_image()
y=predict(network,x)
p=np.argmax(y)
print(p)
print(y)
'''
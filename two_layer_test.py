import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from two_layer_net import TwoLayerNet
from PIL import Image

def init_network():
    with open("two_layer_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def show(img):
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def predict_image(path_to_image):
    im = Image.open(path_to_image)
    im = im.convert('L')

# 转换图像到mnist的大小28*28
    im = im.resize((28,28))
# 获取图像长宽
    (width, height) = im.size
# 将图像数据转化位mnist
    data_image=np.zeros(784)
    
    count=0
    for x in range(0,width):
        for y in range(0,height):
            data_image[count]=(255 -im.getpixel((y,x)))
            count+=1

    network=init_network()
    y=network.predict(data_image)
    
    p=np.argmax(y)
    show(data_image)
    return p

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def get_accuraccy():
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = network.predict(x[i])
        p= np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    return str(float(accuracy_cnt) / len(x))


if __name__ == '__main__':
    
    print("Accuracy:" + get_accuraccy())

    print(predict_image("testdata/2.jpg"))
import numpy as np
import struct
import matplotlib.pyplot as plt
import os

class DataUtils(object):
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'  # 一个I代表一个无符号整数
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def getImage(self):
        # 将MNIST的二进制文件转换城像素特征数据'
        binary_file = open(self._filename, 'rb')
        buf = binary_file.read()
        binary_file.close()
        index = 0
        # numMagic 幻数，它可以用来标记文件或者协议的格式，很多文件都有幻数标志来表明该文件的格式
        # numImgs 图片数
        # numRows, numCols 像素
        numMagic, numImgs, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j]>1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)


    def getLabel(self):
        # 将MNIST中label二进制文件转换成对应的label数字特征
        binary_file = open(self._filename, 'rb')
        buf = binary_file.read()
        binary_file.close()
        index = 0
        magic, numItems = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY):
        # 根据生成的特征和数字标号，输出png的图像
        m, n = np.shape(arrX)
        # 每张是28*28=784Byte
        for i in range(1):
            img = np.array(arrX[i])
            img = img.reshape(28, 28)
            outfile = str(i) + '_' + str(arrY[i]) + '.png'
            plt.figure()
            plt.imshow(img, cmap='binary') # 将图像黑白显示
            plt.savefig(self._outpath + '/' + outfile)


def get_data(train_vs_test=0.3):
    trainfile_x = '../dataset/t10k-images-idx3-ubyte'
    trainfile_y = '../dataset/t10k-labels-idx1-ubyte'
    testfile_x = '../dataset/train-images-idx3-ubyte'
    testfile_y = '../dataset/train-labels-idx1-ubyte'
    data_x = DataUtils(filename=trainfile_x).getImage()
    data_y = DataUtils(filename=trainfile_y).getLabel()
    test_size = int(train_vs_test*len(data_x))
    test_x = data_x[0:test_size, :]
    test_y = data_y[0:test_size]
    train_x = data_x[test_size:, :]
    train_y = data_y[test_size:]
    vali_x = DataUtils(filename=testfile_x).getImage()
    vali_y = DataUtils(filename=testfile_y).getLabel()
    train_data = []
    test_data = []
    vali_data = []
    for x, y in zip(train_x, train_y):
        x = x.reshape((784, 1))
        num = y
        y = np.zeros((10, 1))
        y[num] = 1
        train_data.append((x, y))
    for x, y in zip(test_x, test_y):
        x = x.reshape((784, 1))
        test_data.append((x, y))
    for x, y in zip(vali_x, vali_y):
        x = x.reshape((784, 1))
        vali_data.append((x, y))
    return train_data, test_data, vali_data


if __name__ == '__main__':
    get_data()


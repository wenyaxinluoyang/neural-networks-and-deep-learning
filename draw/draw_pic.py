# encoding=utf-8
# 绘图
import matplotlib.pyplot as plt
import numpy as np

def draw_cost(x1, y1, x2, y2):
    p1, = plt.plot(x1, y1, marker='*', ms=1, color='blue')  # 训练集的代价函数值
    p2, = plt.plot(x2, y2, marker='.', ms=1, color='orange')  # 测试集的代价函数值
    plt.xticks(rotation=45)
    plt.xlabel('round')
    plt.ylabel('cost')
    plt.title('train_dataset & test_dataset cost')
    plt.legend([p1, p2], ['train_dataset', 'test_dataset'], loc='upper right')
    plt.show()


def draw_accuracy(x1, y1, x2, y2):
    p1, = plt.plot(x1, y1, marker='*', ms=1, color='blue') # 训练集的准确率
    p2, = plt.plot(x2, y2, marker='.', ms=1, color='orange') # 测试集的准确率
    plt.xticks(rotation=45)
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.title('train_dataset & test_dataset accuracy')
    plt.legend([p1, p2], ['train_dataset', 'test_dataset'], loc='upper right')
    plt.show()
'''network2.py
使用交叉墒作为代价函数
'''

import json
import random
import sys
import numpy as np
import data_util

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# sigmoid 函数的导数
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# 定义代价函数
# 二次代价函数 Cost = 1/2[(a-y)*(a-y)]
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y)*sigmoid_prime(z)

# 以交叉墒作为代价函数
class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes) # 神经网络的层数
        self.sizes = sizes # 每层神经元的个数
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        '''
        前向传播
        :param a: 输入
        :return:
        '''
        activations = [a]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations, zs

    # training_data 训练集续剧, epochs迭代次数，mini_batch_size每个子样本的个数， eta学习速率
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0,
            evaluation_data=None, monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False, monitor_training_cost=False,
            monitor_training_accuracy=False):
            if evaluation_data: n_data = len(evaluation_data)
            n = len(training_data)
            evaluation_cost, evaluation_accuracy = [], []
            training_cost, training_accuracy = [], []
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
                print('Epoch %s training complete' % j)
                if monitor_training_cost:
                    cost = self.total_cost(training_data, lmbda)
                    training_cost.append(cost)
                    print('Cost on training data: {}'.format(cost))
                if monitor_training_accuracy: # 训练集准确率
                    accuracy = self.accuracy(training_data, convert=True)
                    training_accuracy.append(accuracy/n)
                    print('Accuracy on training data: {}/{}={}'.format(accuracy, n, accuracy/n))
                if monitor_evaluation_cost:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print('Cost on evalutation data: {}'.format(cost))
                if monitor_evaluation_accuracy:
                    accuracy = self.accuracy(evaluation_data)
                    evaluation_accuracy.append(accuracy/n_data)
                    print('Accuracy on evaluation data: {}/{}={}'.format(accuracy, n_data, accuracy/n_data))
            return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases] # 存放b的变化
        nabla_w = [np.zeros(w.shape) for w in self.weights] # 存放w的变化
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda*n))*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, nabla_b)]



    def backprop(self, x, y):
        '''
        反向传播算法
        :param x: 样本
        :param y: 输出
        :return:
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 正向传播，求出每层的输出和带权输出
        activations, zs = self.feedforward(x) # zs为各层的带权输出，activations是各层的输出
        # 开始反向传播
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x,y in data:
            zs, activations = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(activations[-1],y)/len(data) # 计算代价
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def accuracy(self, data, convert=False):
        # 需要转化，说明y是一个向量
        if convert:
            results = [(np.argmax(self.feedforward(x)[0][-1]), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)[0][-1]), y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)


# 如果输入的y是一个数字，把它转化成向量
def vectorized_result(y):
    v = np.zeros((10,1))
    v[y] = 1
    return v


if __name__ == '__main__':
    train_x, train_y, test_x, test_y, vali_x, vali_y = data_util.get_data()
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

    net = Network([784, 30, 10], cost=CrossEntropyCost)
    net.large_weight_initializer()
    train_cost, train_accuracy, evaluation_cost, evaluation_accuracy = net.SGD(train_data, 30, 10 ,1, evaluation_data=test_data,
    monitor_training_cost=True, monitor_training_accuracy=True, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)
    print('=========')
    for cost, accuracy in zip(train_cost, train_accuracy):
        print('训练集=== cost:', cost, ' accuracy:', accuracy)
    print('=========')
    for cost, accuracy in zip(evaluation_cost, evaluation_accuracy):
        print('测试集=== cost:', cost, ' accuracy:', accuracy)






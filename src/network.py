import numpy as np
import random
import data_util


# 使用我们的网络来分类数字
# 激活函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))  # sigmoid = 1/(1+e^-z)
    # 当z是一个向量或是一个numpy数组时，numpy自动地按元素应用sigmoid函数


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)  # sizes是一个list，存放各层神经元的个数, len(sizes) 就是神经网络的层数
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # random.randn函数生成均值为0，标准差为1的高斯分布。
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # 将前一层的神经元个数与后一层的神经元个数对应起来

    def feedforward(self, a):
        # 对于网络给定一个输入a，返回对应的输出
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        # w为matrix(j,i) w是j行i列的矩阵，j代表后一行的神经元个数，i代表前一行的神经元个数
        # a为matrix(i,1) a是i行1列的列向量，是输入向量
        return a

    # 随即梯度下降算法SGD
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''

        :param training_data: 训练输入和其对应的期望输出
        :param epochs: 迭代期数量
        :param mini_batch_size: 采样时小批量数据的大小
        :param eta: 学习速率
        :param test_data: 训练集
        :return:
        '''
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)  # 训练集样本数
        for j in range(epochs):
            random.shuffle(training_data)
            # 每份子样本有mini_batch_size条数据
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 对于每个样本，我们进行一次梯度下降
            aver_correct_rate = 0
            cnt = 0
            if test_data:
                temp = self.evaluate(test_data)
                aver_correct_rate += (temp / n_test)
                cnt = cnt + 1
                print('Epoch {0}: {1} / {2}'.format(j, temp, n_test))
            else:
                print('Epoch {0} complete'.format(j))
        print('平均正确率: ', aver_correct_rate / cnt)

    # 进行梯度下降的算法
    def update_mini_batch(self, mini_batch, eta):
        '''
        通过梯度下降算法不断更新权重和偏置
        :param mini_batch:
        :param eta:
        :return:
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # backprop反向传播算法，计算梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # b + delta(b)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  # w + delta(w)
            self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    # 反向传播算法，计算梯度, 在此使用二次代价函数。
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]  # 放置输入信息
        zs = []  # zs存放每一层的带权输出
        # 前向传播，z = w*a+b  输出 a = sigmoid(z)
        for b, w in zip(self.biases, self.weights):
            temp = np.dot(w, activation)  # w*a
            z = temp + b  # 求某层的输出,同时是下层的输入, z = w*a + b
            zs.append(z)
            activation = sigmoid(z)  # 该层输出 output = sigmoid(z)
            activations.append(activation)  # 添加该层输出作为下一层的输入

        # 反向传播算法
        # C = 1／2*(y-a)^2   ∂C/∂z = (∂C/∂a) * (∂C/∂z) = (a-y)*sigmoid_prime(z)
        # 计算z的误差
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # ∂C/∂b = ∂C/∂z  所以b的误差与c的误差相同
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 计算wjk的调整
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  # 第l层的误差可以用第l+1层的误差求得
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        print(test_results)
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


'''
sigmoid(x) = 1/(1+e^-x)
g(x) = e^-x g(x)的导数：-e^-x
sigmoid(x)的导数 = d/dx
          = [0*（1+e^-x)-1*(-e^-x)]/(1+e^-x)*(1+e^-x) 
          = e^-x / (1+e^-x)*(1+e^-x) = 1/(1+e^-x) * e^-x/(1+e^-x)
          = sigmoid(x) * e^(-x)/(1+e^-x)
          = sigmoid(x) * (1-1/(1+e^-x))
          = sigmoid(x) * (1-sigmoid(x)

sigmoid_prime函数对sigmoid函数进行求导
'''


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


# 使用一个三层神经网络来识别单个数字
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
    net = Network([784, 30, 10])
    # 学习超过30次迭代期, 小批量数据大小为10，学习速率3.0
    net.SGD(train_data, 40, 10, 1, test_data=test_data)
    print('检验验证集')
    temp = net.evaluate(vali_data)
    count = 0
    print('验证集平均正确率:', temp / len(vali_data))
    # 隐藏层节点30,迭代期40,学习速率2,正确率(平均)0.834
    # 隐藏层节点20,正确率(平均)0.81
    # 隐藏层节点20,学习速率4,正确率0.80
    # 隐藏层节点30,迭代期40,学习速率1,正确率(平均)0.88




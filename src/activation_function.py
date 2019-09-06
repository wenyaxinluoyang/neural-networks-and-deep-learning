# 封装各种激活函数
import numpy as np

class SigmoidFun(object):
    @staticmethod
    def fn(z):  # 函数形式: 1/(1+e^-x)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def fn_prime(z):
        return SigmoidFun.fn(z) * (1 - SigmoidFun.fn(z))


class SoftmaxFun(object):
    '''
    柔性最大值:
    '''
    @staticmethod
    def fn(z):
        fm = sum([np.exp(x[0]) for x in z])
        result = np.array([[np.exp(x[0])/fm] for x in z])
        return result

    @staticmethod
    def fn_prime(self, z):
        pass


def test(array):
    output = SoftmaxFun.fn(array) # 使用柔性极大值作为激活函数求输出
    return output

def test_two(array):
    array[3] = 2 # 使第四个值增大
    output = SoftmaxFun.fn(array) # 输出结果中，第四个值增大，其他值都减小
    return output

def test_three(array):
    array[3] = 5 # 继续使第四个值增大
    output = SoftmaxFun.fn(array) # 输出结果中，第四个值增大，其他值都减小
    return output

def test_four(array):
    array[3] = -2 # 第四个值减小
    output = SoftmaxFun.fn(array) # 输出结果中，第四个值减少，其他值都增大
    return output

def test_five(array):
    array[3] = -5
    output = SoftmaxFun.fn(array) # 输出结果中，第四个值减少，其他值都增大
    return output


if __name__ == '__main__':
    array = np.zeros((4, 1))
    array[0] = 2.5
    array[1] = -1
    array[2] = 3.2
    array[3] = 0.5
    print(array)
    print('========')
    print(test(array))
    print('========')
    print(test_two(array))
    print('=======')
    print(test_three(array))
    print('**********************')
    array[3] = 0.5
    print(test(array))
    print('=======')
    print(test_four(array))
    print('=======')
    print(test_five(array))


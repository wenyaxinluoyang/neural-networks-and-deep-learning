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


def test():
    array = np.zeros((4,1))
    array[0] = 2.5
    array[1] = -1
    array[2] = 3.2
    array[3] = 0.5
    print(array)
    print('================')
    output = SoftmaxFun.fun(array)
    print(output)

if __name__ == '__main__':
   test()


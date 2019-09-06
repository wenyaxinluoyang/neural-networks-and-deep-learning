# 代价函数
import numpy as np
from activation_function import SigmoidFun

# 二次代价函数
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * SigmoidFun.sigmoid_prime(z)


# 交叉墒代价函数
# 以交叉墒作为代价函数
class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)





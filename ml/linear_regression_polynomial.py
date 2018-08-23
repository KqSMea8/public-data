#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线性回归
简单参数 y=ax**2+bx+c ?
参考：


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def generate_dataset():
    """生成训练数据 y=ax+b"""
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    # 在指定的间隔内返回均匀间隔的数字
    train_X = np.linspace(-1, 1, 100)
    # np.random.randn(*train_X.shape) * 0.33 添加随机抖动
    train_Y = 2 * (train_X**2) + np.random.randn(*train_X.shape) * 0.33 + 10
    # print(train_X)
    # print(train_Y)
    # print(np.random.randn(*train_X.shape)* 0.33)
    return train_X, train_Y


def draw_pic(X, Y, m, b):
    a=[]
    b=[]
    # y=0
    # x=-50

    for x in range(-50,50,1):
        y=x**2+2*x+2
        a.append(x)
        b.append(y)
        #x= x+1

    y = 2 * (X**2) + 10
    
    fig= plt.figure()
    plt.scatter(X, Y, alpha=0.8)
    
    axes=fig.add_subplot(111)
    axes.plot(X,y)

    axes1=fig.add_subplot(111)
    y1 = m * (X**2) + b
    axes1.plot(X,y1)

    plt.show() 
    


class Linear_regression:
    """线性回归,需参考sklearn的实现"""

    def __init__(self, W, b, learning_rate=0.001, num_iter=20000):
        self.W = W
        self.b = b
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.N = 0

    def predict(self, X):
        """定义预测模型 y=ax+b"""
        return self.W*(X**2) + self.b  # +，- b不影响？

    def __loss(self, X, Y):
        """定义损失函数:均方误差mse"""
        return np.sqrt(((Y - self.predict(X)) ** 2).mean())
        # return np.sqrt((( self.predict(X) - Y ) ** 2).mean()) 
        # return np.sum((Y - self.predict(X))**2, axis=0)/self.N

    def __gradient_descent_optimizer(self, X, Y):
        """优化算法:梯度下降"""
        # error = Y - (self.W*X+self.b)
        error = Y - self.predict(X) # Y - pre_Y 这个顺序不能换？
        # error = self.predict(X) - Y  

        w_gradient = -(2/self.N)*error*X  # self.predict(X)？ x又是什么鬼？
        w_gradient = np.sum(w_gradient, axis=0) 
        b_gradient = -(2/self.N)*error  # 均方误差求导
        b_gradient = np.sum(b_gradient, axis=0)

        #去掉-(2/self.N)这样的常数，似乎对优化结果无影响
        # w_gradient = -error*X  # self.predict(X)？ x又是什么鬼？
        # w_gradient = np.sum(w_gradient, axis=0) 
        # b_gradient = -error  # 均方误差求导
        # b_gradient = np.sum(b_gradient, axis=0)

        self.W = self.W - (self.learning_rate * w_gradient)
        self.b = self.b - (self.learning_rate * b_gradient)

    def fit(self, X, Y):
        """训练 train"""
        self.N = float(len(X))

        for i in range(self.num_iter):
            self.__gradient_descent_optimizer(X, Y)
            if i % 100 == 0:
                error = self.__loss(X, Y)
                print('iter{0}:W={1},b={2},error={3}'.format(
                    i, self.W, self.b, error))

        return self.W, self.b


if __name__ == "__main__":
    x, y = generate_dataset()
    model = Linear_regression(0.0, 0.0, 0.001, 20000)
    w, b = model.fit(x, y)
    print(w, b)
    print('x=6.6, y=', model.predict(6.6))
    draw_pic(x, y, w, b)

    # model

    # draw_pic(x,y)
    # pass

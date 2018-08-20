#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多(1+)特征 线性回归
https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def generate_dataset():
    """生成训练数据 y=ax+b""" 
    # W = np.array([[2], [5], [10]])
    W = np.array([2.0,3.0,10.0]).reshape(-1, 1) 

    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    train_X = np.linspace(-1, 1,100).reshape(-1, 2)

    #train_Y
    tmp_x = np.hstack([train_X, np.ones((train_X.shape[0], 1))])  
    tmp_y = tmp_x.dot(W) 
    tmp_y = tmp_y + np.random.randn(*tmp_y.shape) * 0.33 #加入随机抖动
    train_Y = tmp_y.reshape(-1, 1)  
     
    return train_X, train_Y


def draw_pic(X, Y, m, b):
    plt.scatter(x, y, alpha=0.8)
    plt.plot([-1, 1], [-1*2+10, 1*2+10], ls='solid')  # 实际直线
    plt.plot([-1, 1], [-1*m+b, 1*m+b], ls='dashdot')  # 拟合直线
    plt.grid(True)
    plt.show()
    # pass


class Linear_regression:
    """线性回归,需参考sklearn的实现"""

    def __init__(self, W, learning_rate=0.001, num_iter=20000):
        self.W = W
        # self.b = b
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.N = 0

    def predict(self, X):
        """定义预测模型 y=ax+b"""
        return X.dot(self.W) 

    def __loss(self, X, Y):
        """定义损失函数:均方误差mse
        rmse：(根均方误差)
        mse：(均方误差) 
        """  
        return (np.sum((self.predict(X) - Y)**2)) / (2*self.N) 

        # tobesummed = np.power(((X @ theta.T)-y),2)
        # return np.sum(tobesummed)/(2 * len(X))  

        # J2 = (C.T.dot(C)) / (2*self.N)
        # return J2  

    def __gradient_descent_optimizer(self, X, Y):
        """优化算法:梯度下降""" 
        # https://blog.csdn.net/qq_26222859/article/details/73326088
        # https://uqer.io/v3/community/share/596da6e8f83a2100527016b0

        # theta = self.W
        # temp = np.matrix(np.zeros(theta.shape))
        # parameters = int(theta.shape[1]) #ravel().
        # # cost = np.zeros(iters)
        
        # # for i in range(iters):
        # error = (X * theta.T) - Y
        
        # for j in range(parameters):
        #     term = np.multiply(error, X[:,j])
        #     temp[0,j] = theta[0,j] - ((self.learning_rate / len(X)) * np.sum(term))
            
        # self.W = temp
        
            # cost[i] = computeCost(X, y, theta)
            
        # return theta, cost

        # parameters = int(self.W.ravel().shape[1])
        # for j in range(parameters):
        #     term = np.multiply(error, X[:,j])

        # theta = self.W
        # theta = theta - (self.learning_rate/len(X)) * np.sum(X * (X @ theta.T - Y), axis=0)

        theta = self.W - (self.learning_rate/self.N) * \
            (X.T.dot(X.dot(self.W) - Y)) 
        self.W = theta

    def fit(self, X, Y):
        """训练 train"""
        self.N = Y.shape[0]  # float(len(X))

        X = np.hstack([X, np.ones((x.shape[0], 1))])

        for i in range(self.num_iter):
            self.__gradient_descent_optimizer(X, Y)
            if i % 100 == 0:
                error = self.__loss(X, Y)
                print('iter{0}:W={1},error={2}'.format(
                    i, self.W,  error))

        return self.W


if __name__ == "__main__":
    x, y = generate_dataset()
    # print(x)
    # print(y)
    model = Linear_regression(np.zeros((3, 1)), 0.001, 100000)
    w = model.fit(x, y)
    print(w)
    # print('x=[6.6,7], y=', model.predict([6.6,7]))
    # draw_pic(x, y, w, b)

    # model

    # draw_pic(x,y)
    # pass

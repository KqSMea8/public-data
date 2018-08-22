#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多元线性回归，矩阵运算
https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905
https://github.com/Tan-Moy/medium_articles/blob/master/art2_multivariate_linear_regression/mlr.py

http://zhouyichu.com/machine-learning/Gradient-Code/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import matplotlib.pyplot as plt


def gen_train_data(w, row_number=5):
    """生成训练数据"""
    W = np.array(w)  # .reshape(-1,1)
    # print(W.reshape(-1,1))

    dim_number = W.shape[0]-1  # W的最后一项是偏置项
    # print(dim_number)

    train_X = np.linspace(-1, 1, dim_number *
                          row_number).reshape(-1, dim_number)  # 二维度数组

    # print(np.ones((train_X.shape[0], 1)))
    tmp_x = np.hstack([train_X, np.ones((train_X.shape[0], 1))])  # x 最后一列都是1
    # print(tmp_x)

    train_Y = tmp_x @ W.T
    # print(tmp_y)
    r = np.random.randn(*train_Y.shape) * 0.33
    # print( r )
    train_Y = train_Y + r  # 加入数值随机抖动 .reshape(-1,1)

    return train_X, train_Y.reshape(-1, 1)


class Linear_regression:
    """线性回归,需参考sklearn的实现"""

    def __init__(self, w, learning_rate=0.001, num_iter=20000):
        self.W = np.array(w)
        # print(self.W)

        self.learning_rate = learning_rate
        self.num_iter = num_iter

    def predict(self, X):
        """定义预测模型 y=ax+b"""
        X = np.hstack([X, np.ones((x.shape[0], 1))]
                      )  # 考虑到偏置项b参与到矩阵运算，X最后加一个值为1的列
        return X @ self.W.T  # 不等价于 X.dot(self.W)
        # return X.dot(self.W)

    def loss(self, X, Y):
        """定义损失函数:均方误差mse
        rmse：(根均方误差)
        mse：(均方误差) 
        """
        cost = np.sum((self.predict(X) - Y)**2) / (2*X.shape[0])
        # print("cost1", cost)

        # #np.power 和 **2 二者等价 ，哪个更好些？
        # cost = np.sum(np.power( self.predict(X)-Y, 2)) / (2*X.shape[0])
        # print("cost2",cost)

        return cost

    def optimize(self, X, Y):
        """优化算法:梯度下降"""
        # https://blog.csdn.net/qq_26222859/article/details/73326088
        # https://uqer.io/v3/community/share/596da6e8f83a2100527016b0

        X = np.hstack([X, np.ones((X.shape[0], 1))])

        theta = self.W - (self.learning_rate/len(X)) * np.sum(X * (X @ self.W.T - Y), axis=0)
        self.W = theta

        # theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        # print("X @ self.W.T =",  X @ self.W.T )
        # print("X @ self.W.T - Y =", X @ self.W.T - Y)
        # print("X * ( X @ self.W.T - Y ) =",  X * ( X @ self.W.T - Y )  ) #error ? ValueError: operands could not be broadcast together with shapes (5,3) (5,)
        # print( np.sum(X * ( X @ self.W.T - Y ), axis=0)  )

        # tmp_x = np.hstack([X, np.ones((X.shape[0], 1))])
        # theta = self.W - (self.learning_rate/len(tmp_x)) * np.sum(tmp_x * ( self.predict(X) ), axis=0)

    def fit(self, X, Y):
        """训练 train"""
        # X = np.hstack([X, np.ones((x.shape[0], 1))])

        for i in range(self.num_iter):
            self.optimize(X, Y)
            if i % 100 == 0:
                error = self.loss(X, Y)
                print('iter{0}:W={1},error={2}'.format(
                    i, self.W,  error))

        return self.W


if __name__ == "__main__":
    w = [1, 3, 5]
    x, y = gen_train_data(w, 500)

    learning_rate = 0.001
    num_iter = 10000
    # print(x)
    # print(y)
    # print(x.shape[0])
    # print(w.shape)
    # print(np.zeros(len(w)))

    # print(np.zeros([1,3]))

    # np.zeros((3,5))
    model = Linear_regression(np.zeros([1, len(w)]), learning_rate, num_iter)
    w = model.fit(x, y)
    # # # pre_y = model.predict(x)
    # # # print(pre_y)

    # cost = model.loss(x, y)

    # print(w)
    # # print('x=[6.6,7], y=', model.predict([6.6,7]))
    # # draw_pic(x, y, w, b)

    

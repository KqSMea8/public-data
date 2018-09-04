#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

"""
参考：
https://beckernick.github.io/logistic-regression-from-scratch/

"""


def gen_train_data(w, row_number=5):
    """生成训练数据"""
    pass


def show_sigmoid():
    nums = np.arange(-10, 10, step=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nums, sigmoid(nums), 'r')
    plt.show()


def show(data):
    """https://zhuanlan.zhihu.com/p/31954955"""
    

    # how 打乱顺序?
    # data.reindex(np.random.permutation(data.index))
    # data.apply(np.random.shuffle)

    print(data.head())

    # print(data['label'].isin([1]))
    positive = data[data['label'].isin([1])]  # label 列中的1设定为positive
    negative = data[data['label'].isin([0])]  # label 列中的0设定为negative

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(positive['x'], positive['y'], s=50,
               c='b', marker='o', label='yes')
    ax.scatter(negative['x'], negative['y'], s=50,
               c='r', marker='x', label='no')

    ax.legend()
    ax.set_xlabel('x Score')
    ax.set_ylabel('y Score')
    plt.show()


class Logistic_Regression():
    def __init__(self, theta, alpha, iter_num):
        self.theta = np.matrix(theta)
        self.alpha = alpha
        self.iter_num = iter_num

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict_probability(self, X):
        return self.sigmoid(X * self.theta.T)

    def predict(self,  X):
        probability = self.sigmoid(X * self.theta.T)
        return [1 if x >= 0.5 else 0 for x in probability]

    def loss(self, x, y):
        # theta = np.matrix(theta)
        # X = np.matrix(X)
        # y = np.matrix(y)
        first = np.multiply(-y, np.log(self.sigmoid(x * self.theta.T)))
        second = np.multiply(
            (1 - y), np.log(1 - self.sigmoid(x * self.theta.T)))
        return np.sum(first - second) / (len(x))

    def optimize(self, x, y):
        # theta = np.matrix(theta)

        #梯度上升？

        # theta = self.theta - (self.alpha/len(X)) * np.sum(X * ( self.predict(X) - y), axis=0)
        # self.theta = theta

        h = self.sigmoid(x*self.theta.T)
        # print(h)
        # print(y)
        error = (y - h)
        self.theta = self.theta + self.alpha * x.T * error

        # h = self.sigmoid(X*self.theta.T)
        # error = (y - h)
        # self.theta = self.theta + np.matrix(self.alpha * X.transpose()* error )

        # dataMat=mat(dataArray)    #size:m*n
        # labelMat=mat(labelArray)      #size:m*1
        # m,n=shape(dataMat)
        # weigh=ones((n,1))
        # for i in range(maxCycles):
        #     h=sigmoid(dataMat*weigh)
        #     error=labelMat-h    #size:m*1
        #     weigh=weigh+alpha*dataMat.transpose()*error
        # return weigh

        # parameters = int(self.theta.ravel().shape[1])
        # grad = np.zeros(parameters)

        # error = self.sigmoid(X * self.theta.T) - y

        # for i in range(parameters):
        #     term = np.multiply(error, X[:, i])
        #     grad[i] = np.sum(term) / len(X)

        # self.theta = self.theta - self.alpha*np.matrix(grad)
        # return self.theta

    def fit(self, x, y):
        x = np.matrix(x)
        y = np.matrix(y)

        for i in range(self.iter_num):
            self.optimize(x, y)
            if i % 100 == 0:
                error = self.loss(x, y)
                print('iter{0}:W={1},error={2}'.format(
                    i, self.theta, error))


def run():
    path = 'open_source/ex2data1.txt'  # 路径要设置为你自己的路径
    data = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    show(data)

    # add a ones column - this makes the matrix multiplication work out easier
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols-1]
    y = data.iloc[:, cols-1:cols]

    # convert to numpy arrays and initalize the parameter array theta
    X = np.array(X.values)
    y = np.array(y.values)

    theta = np.zeros(3)
    alpha = 0.0001
    iter_num = 20000

    model = Logistic_Regression(theta, alpha, iter_num)
    tmp = model.fit(X, y)
    print(tmp)

    # result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    # print(result)

    # gradient(theta, X, y)


if __name__ == "__main__":
    # show()
    # show_sigmoid()
    run()

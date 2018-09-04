#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://beckernick.github.io/logistic-regression-from-scratch/

https://github.com/shevonwang/Python-Logistic-Regression/blob/master/main.py

"""
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


def generate_data():
    """生成数据"""
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal(
        [0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal(
        [1, 4], [[1, .75], [.75, 1]], num_observations)  

    # 加入随机抖动？
    
    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))

    # print(simulated_separableish_features)

    return simulated_separableish_features, simulated_labels


def show(features, labels, weights):
    """图表展示"""
    plt.figure(figsize=(12, 8))
    plt.scatter(features[:, 0], features[:, 1],
                c=labels, alpha=.4)

    # 绘制决策边界？
    # func = lambda x: (1-weights[2]-weights[0]*x)/weights[1] #err a1x+a2y+a3=1
    def func(x): return (1-weights[0]-weights[1]*x)/weights[2]  # a1x+a2y+a3=1
    x = np.array([-4.5, 4.5])  # 使用numpy批量乘
    y = func(x)
    plt.plot(x, y)
    plt.show()


def sigmoid(scores):
    """sigmoid"""
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    """对数最大似然 计算cost"""
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=True):
    """逻辑回归"""
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))  # ?
        # print(features)

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal) #梯度公式？
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 100 == 0:
            cost = log_likelihood(features, target, weights)

            # Accuracy
            preds = np.round(predictions)
            acc = (preds == target).sum().astype(float) / len(preds)

            print('iter{0}:cost={1},accuracy={2},W={3}'.format(
                step, cost, acc, weights))

    return weights


features, labels = generate_data()
# weights=[-14.09223558,  -5.0589895  ,  8.28954618]
weights = logistic_regression(
    features, labels, num_steps=3000, learning_rate=5e-5)
show(features, labels, weights)

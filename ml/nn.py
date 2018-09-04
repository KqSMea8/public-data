#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network.ipynb
https://beckernick.github.io/neural-network-scratch/


https://python.freelycode.com/contribution/detail/300
https://www.jianshu.com/p/679e390f24bb
https://blog.csdn.net/oxuzhenyi/article/details/73026790

https://www.leiphone.com/news/201806/mHFSo8zyLsX7L581.html #无github代码？

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def gen_data():
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[2, .75],[.75, 2]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    x3 = np.random.multivariate_normal([2, 8], [[0, .75],[.75, 0]], num_observations)
    
    #RuntimeWarning: covariance is not positive-semidefinite.
    #x3 = np.random.multivariate_normal([2, 8], [[0, .75],[.75, 0.001]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2, x3)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                    np.ones(num_observations), np.ones(num_observations) + 1))
    return simulated_separableish_features,simulated_labels

def show_fig(simulated_separableish_features,simulated_labels):
    plt.figure(figsize=(12,8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c = simulated_labels, alpha = .4)




class NeuralNetwork: 
    def __init__(self, x, y): 
        self.input      = x 
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)  
        self.y          = y 
        self.output     = np.zeros(self.y.shape) 
    
    def sigmoid(self,val):
        return 1/(1+np.exp(val))

    def feedforward(self): 
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1)) 
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2)) 

    def backprop(self): 
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output))) 
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
 
        # update the weights with the derivative (slope) of the loss function 
        self.weights1 += d_weights1 
        self.weights2 += d_weights2



if __name__ == "__main__":
    # line9()

    x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    nn = NeuralNetwork(x,y)
    nn.feedforward()
    nn.backprop()
    

    # features,labels = gen_data()
    # show_fig(features,labels)
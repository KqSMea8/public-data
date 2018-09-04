#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nn xor 异或
https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(val):
    return 1 / (1 + np.exp(-val)) 

def predict(weights):
    test_features = np.array([[1, 0, 0],[0, 1, 0],[0, 1, 0],[0,0,0]])
    score = 1 / (1 + np.exp(-(np.dot(test_features, weights))))
    # label = 1 if score[0] >= 0.5 else 0
    return score 

def line9():
    """
    xor: 都相同=1，不同=0， 明显不是？
    0 0 1 - 0
    0 1 1 - 0
    1 1 1 - 1
    1 0 1 - 1

    0 1 0  
    1 0 0
    1 1 0
    0 0 0  
    """
    #from numpy import exp, array, random, dot
    features = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    labels = np.array([[0, 1, 1, 0]]).T

    np.random.seed(1)
    weights = 2 * np.random.random((3, 1)) - 1

    for iteration in range(10000):
        output = 1 / (1 + np.exp(-(np.dot(features, weights)))) #前向
        weights += np.dot(features.T, (labels - output) #反向
                          * output * (1 - output))
        if iteration % 100 == 0:
            score = predict(weights)
            print("iter=%s,weights=%s,score=%s" % (iteration, weights.tolist(),score.tolist()))

    

if __name__ == "__main__":
    line9()

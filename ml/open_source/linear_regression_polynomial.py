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
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def residual(t, x, y):
    return y - (t[0] * x ** 2 + t[1] * x + t[2])

x = np.linspace(-2, 2, 50)
A, B, C = 2, 3, -1           #为真实值
y = (A * x ** 2 + B * x + C) + np.random.rand(len(x))*0.75  

p = leastsq(residual, [0, 0, 0], args=(x, y))

theta = p[0] #将拟合出来的参数赋值给theta
print('真实值：', A, B, C)
print('预测值：', theta)
y_hat = theta[0] * x ** 2 + theta[1] * x + theta[2]
plt.plot(x, y, 'r-', linewidth=2, label=u'Actual')
plt.plot(x, y_hat, 'g-', linewidth=2, label=u'Predict')
plt.legend(loc='upper left')
plt.grid()
plt.show()
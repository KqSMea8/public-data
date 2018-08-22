#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

"""
基于numpy的各种矩阵操作
https://blog.csdn.net/tintinetmilou/article/details/78587760 矩阵求导
"""



def gen_train_data(w, b=0.3, row_number=5):
    W = np.array(w)
    dim_number = W.shape[0]

    # print(dim_number)
    # print(W.shape)

    # 生成X
    train_X = np.linspace(-1, 1, dim_number *
                          row_number).reshape(-1, dim_number)  # 二维度数组
    # print(train_X)

    # 生成Y
    # y = a0*x0 + a1*x1 + b 的矩阵表示？

    # print(W*train_X) #点乘,非预期
    # print(train_X*W) #点乘

    # print(W @ train_X) #ERR
    # print(train_X @ W.T)
    tmp_y = train_X @ W.T + b  # 矩阵乘法,加入偏置
    # print(tmp_y)
    r = np.random.randn(*tmp_y.shape) * 0.33  # 加入数值抖动
    # print( r )
    train_Y = (tmp_y + r)  # .reshape(-1,1)
    return train_X, train_Y


def gen_train_data_v2(w, row_number=5):
    W = np.array(w)

    dim_number = W.shape[0]-1  # W的最后一项是偏置项

    train_X = np.linspace(-1, 1, dim_number *
                          row_number).reshape(-1, dim_number)  # 二维度数组

    # print(np.ones((train_X.shape[0], 1)))
    tmp_x = np.hstack([train_X, np.ones((train_X.shape[0], 1))])  # x 最后一列都是1

    tmp_y = tmp_x @ W.T
    # print(tmp_y)
    r = np.random.randn(*tmp_y.shape) * 0.33  # 加入数值抖动
    # print( r )

    train_Y = (tmp_y + r)  # .reshape(-1,1)
    return train_X, train_Y


if __name__ == "__main__":
    # train_X,train_Y = gen_train_data([2,3,5,9],2,10)
    train_X, train_Y = gen_train_data_v2([2, 3, 5, 9], 2)
    # print("train_X" , train_X)
    # print("train_Y", train_Y)

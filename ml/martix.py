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

def martix_init():
    """矩阵初始化"""
    zeros = np.zeros(shape=(5,2))
    print(zeros)
    print(zeros.shape)

    ones = np.ones(shape=(5,1)) 
    print(ones)

    print(np.hstack([zeros,ones])) #

def martix_plus():
    """加法"""
    m1 = np.array([[1,2,3],[4,5,6]]) 
   
    print("---矩阵+标量---")
    # assert (m1 + 2)==(2+m1), "(m1 + 2)==(2+m1) not equal" #?
    print(m1+2,2+m1) 
     
    print("---矩阵+向量， 列必须一致---")
    v1 = np.array([1,2,3])
    print(m1+v1,v1+m1) 
    # print(m1+ np.array([1,2,3,4]) ) #error,列必须相同
     
    print("---矩阵+矩阵---") 
    print(m1+np.array([[1,2,3]]))
    print(m1+np.array([[1,2,3],[7,6,5]]))
    print(np.array([[1,2,3],[7,6,5]]) + m1 )
    # print(m1+np.array([[1,2,3],[7,6,5],[3,4,4]])) #ValueError: operands could not be broadcast together with shapes (2,3) (3,3)
    # print(np.array([[1,2,3],[7,6,5],[3,4,4]]) + m1 ) #ValueError: operands could not be broadcast together with shapes (3,3) (2,3)
     

def martix_multiplication():
    """矩阵乘法"""
    m1 = np.array([[1,2,3],[4,5,6]]) 

    print("---矩阵*标量---")  
    print(m1*3)
    print(3*m1) # 交换率,结合律
    print(m1*0)

    print("---矩阵*向量---") 
    v1 = np.array([1,2,3])
    print(m1 * v1) #点乘？
    # print(v1.T)
    print(m1 @ v1) 
    # print(m1 @ v1.T) #m1@v1==m1@v1.T ?
    # print(v1 @ m1) #交换率，error
    # print(v1 @ m1.T) 
    # print(m1*0)

    print("---矩阵*矩阵---")  
    # mn * na
    a = np.array([[1,2],[3,4]]) 
    b = np.array([[11,12],[13,14]]) 
    print(np.dot(a,b))
    print(a@b) 
    print(a*b) #点乘
    
    print("---矩阵转置---")  
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.T.html
    x = np.array([[1.,2.],[3.,4.]])
    print(x.T)
    print(x.transpose())
    x = np.array([1.,2.,3.,4.])
    print(x.T)

    # 单位矩阵
    # 矩阵求逆
    # 公式求解最小值  



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

    tmp_y = tmp_x @ W.T #矩阵乘法 #转置
    # print(tmp_y)
    r = np.random.randn(*tmp_y.shape) * 0.33  # 加入数值抖动
    # print( r )

    train_Y = (tmp_y + r)  # .reshape(-1,1)
    return train_X, train_Y


if __name__ == "__main__":
    # martix_init()
    # martix_plus()
    martix_multiplication()
    # train_X,train_Y = gen_train_data([2,3,5,9],2,10)
    # train_X, train_Y = gen_train_data_v2([2, 3, 5, 9], 2)
    # print("train_X" , train_X)
    # print("train_Y", train_Y)

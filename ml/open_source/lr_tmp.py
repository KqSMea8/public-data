#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def H(theta,x):
    return theta.dot(x)

def Read():
    data = np.genfromtxt('ex1data2_2.txt',delimiter=',') # loadtxt('ex1data2')
    x = data[:,(0,1)].reshape((-1,2))
    y = data[:,2].reshape((-1,1))
    m = y.shape[0]

    return(x,y,m)

    # x = np.loadtxt('ex1data2')
    # y = np.loadtxt('ex3y.dat')
    # m = len(x)
    # t = np.ones((m,1))
    # x = np.concatenate((t,x),axis=1)
    # return (x,y,m)

# def load_exdata(filename):
#     data = []
#     with open(filename, 'r') as f:
#         for line in f.readlines(): 
#             line = line.split(',')
#             current = [int(item) for item in line]
#             #5.5277,9.1302
#             data.append(current)
#     return data

# data = load_exdata('ex1data2_2.txt')
# data = np.array(data,np.int64)

# x = data[:,(0,1)].reshape((-1,2))
# y = data[:,2].reshape((-1,1))
# m = y.shape[0]

#gradient的计算
def cal(alpha,theta,x,y,m):
    n = x.shape[1]
    newtheta = np.array([0]*n,dtype=np.float)
    for j in range(0,n):
        count = 0
        for i in range(m):
            count += (H(theta,x[i,:]) - y[i]) * x[i,j]
        newtheta[j] = (theta[j] - alpha / m * count )
    return newtheta

#Cost Function
def J(theta,x,y,m):
    return np.transpose(x.dot(theta)-y).dot(x.dot(theta)-y)/(2*m)

#Normal Equation
def normalequation(x,y):
    return np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x)).dot(y)

#根据不同的alpha值，画出图像
def gradent_descent():
    (x,y,m) = Read()
    sigma = np.std(x,0)
    mu = np.mean(x,0)
    #数据归一化
    x[:,1] = (x[:,1]-mu[1]) / sigma[1]
    x[:,2] = (x[:,2]-mu[2]) / sigma[2]
    n = x.shape[1]

    theta = np.array([0]*n,dtype=np.float)
    j = []
    alpha=0.01
    for i in range(50):
        j.append(J(theta,x,y,m))
        theta = cal(alpha,theta,x,y,m)
    plt.plot(range(50),j,'b-',label=r'$\alpha = 0.01$')


    j = []
    theta = np.array([0]*n,dtype=np.float)
    alpha=0.03
    for i in range(50):
        j.append(J(theta,x,y,m))
        theta = cal(alpha,theta,x,y,m)
    plt.plot(range(50),j,'r-',label=r'$\alpha = 0.03$')

    j = []
    theta = np.array([0]*n,dtype=np.float)
    alpha=0.1
    for i in range(50):
        j.append(J(theta,x,y,m))
        theta = cal(alpha,theta,x,y,m)
    plt.plot(range(50),j,'y-',label=r'$\alpha = 0.1$')

    j = []
    theta = np.array([0]*n,dtype=np.float)
    alpha=0.3
    for i in range(50):
        j.append(J(theta,x,y,m))
        theta = cal(alpha,theta,x,y,m)
    plt.plot(range(50),j,'b--',label=r'$\alpha = 0.3$')


    j = []
    theta = np.array([0]*n,dtype=np.float)
    alpha=1
    for i in range(50):
        j.append(J(theta,x,y,m))
        theta = cal(alpha,theta,x,y,m)
    plt.plot(range(50),j,'r--',label=r'$\alpha = 1$')

    j = []
    theta = np.array([0]*n,dtype=np.float)
    alpha=1.3
    for i in range(50):
        j.append(J(theta,x,y,m))
        theta = cal(alpha,theta,x,y,m)
    plt.plot(range(50),j,'y--',label=r'$\alpha = 1.3$')


    plt.xlabel('Number of interations')
    plt.ylabel('Cost J')
    plt.legend()
    plt.show()


if __name__=='__main__':
    
    # print (x,y,m)

    #画出图像
    gradent_descent()
    (x,y,m) = Read()
    
    #利用normal equation计算theta值
    theta = (normalequation(x,y))
    #预测未知数据
    # print('predict:',end=' ')
    print(H(theta,np.array([1,1650,3])))
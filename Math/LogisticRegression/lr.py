#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
doc:
https://lz5z.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B8%B8%E7%94%A8%E7%AE%97%E6%B3%95%E2%80%94%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92/
"""
import os
import sys
import numpy as np
import sklearn as skl 
# from sklearn import preprocessing
# from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def wget(source_url, local_file=None):
    """数据下载"""
    if not local_file:
        local_file = source_url.split('/')[-1]
    # local_file = '/tmp/' + local_file
    if not (os.path.isfile(local_file) and os.path.getsize(local_file) != 0):
        os.system("wget '%s' -O %s" % (source_url, local_file))
    return local_file


def main():
    # 加载数据
    local_file = wget(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    )

    # 把CSV文件转化为numpy matrix
    dataset = np.loadtxt(local_file, delimiter=",")
    # 训练集和结果
    x = dataset[:, 0:7] #特征
    y = dataset[:, 8] #label
    # 数据归一化
    normalized_x = skl.preprocessing.normalize(x)

    # 逻辑回归
    model = LogisticRegression() 
    model.fit(normalized_x, y)

    # 预测
    expected = y
    predicted = model.predict(normalized_x)

    # 模型拟合概述
    print(skl.metrics.classification_report(expected, predicted))
    print(skl.metrics.confusion_matrix(expected, predicted))



if __name__ == "__main__":
    main()


'''
问题： /2.7/Extras/lib/python/scipy/sparse/coo.py:200: 
VisibleDeprecationWarning: `rank` is deprecated; 
use the `ndim` attribute or function instead. 
To find the rank of a matrix see `numpy.linalg.matrix_rank`.
解决办法：pip install --upgrade scipy --user -U

http://www.jianshu.com/p/ef37f739b531

'''
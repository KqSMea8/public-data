#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码阅读
https://pytorch.org/docs/0.4.0/_modules/torchvision/datasets/mnist.html#MNIST
"""
from __future__ import print_function
import numpy as np 
from PIL import Image
import torch 
import codecs


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16) #

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049 #?这个数字怎么来的？
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

if __name__ == "__main__":
    # label_data = read_label_file("../data/raw/t10k-labels-idx1-ubyte")
    # print(label_data.shape)
    # for i,v in enumerate(label_data):
    #     print(i,v)
    #     if i>100:break 
    
    img_data = read_image_file("../data/raw/t10k-images-idx3-ubyte")
    print(img_data.shape)
    for i,v in enumerate(img_data):
        print(v.shape)
        print(i,v)

        img = Image.fromarray(v.numpy(), mode='L') #, 'RGB'
        img.save("../data/img_"+str(i)+".jpg")
        img.show() 
        
        
        if i>2:break 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms

import numpy as np
from PIL import Image

from simplecnn import SimpleCNN

model_file = "../data/mnist.pth"
model = SimpleCNN()

# 模型保存&加载的2种方式
# 方案1，只存储模型的各种参数，推荐
# torch.save(model.state_dict(), model_file) #如果是这种方式存储模型
model.load_state_dict(torch.load(model_file))
print("model:", model)

# 方案2，不推荐
# torch.save(model_file) #
# model = torch.load(model_file) #如果存储用1，当前的方式会只加载了词典


model.eval()  # 作用是啥？将模型设置为 evaluation 模式,不用反向传播计算

tfc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,), (0.1081,))  # 修改这个数值似乎对模型的预测能力没影响？
])


def eval_single_img(img_file):
    images = np.array([])
    image = tfc(Image.open(img_file).convert('L'))
    print("image.numpy().shape:", image.numpy().shape)
    images = np.append(images, image.numpy())
    img = images.reshape(-1, 1, 28, 28)  # [batch_size,channels,w,h]
    data = torch.from_numpy(img).float()
    print("data.shape:", data.shape)

    with torch.no_grad():  # 不加这个有什么影响？不加会计算图结构等，浪费内存资源
        # data = data.to(device)
        output = model(data)  # 每个分类的预测得分
        # tensor([[ -74.3974,-68.3368,-41.4392,-51.1502,-68.1614,-81.0373,-116.1543,0.0000,-67.3520,-46.4981]])
        print("output:", output)
        pred = output.max(1, keepdim=True)[1]
        # log_softmax 返回的是所有分类按概率高低的排序？no,是批次的所有值
        return pred[0]


if __name__ == '__main__':
    for i in range(4):
        print(eval_single_img("../data/img_%s.jpg" % (i)))
        print("~~~~")

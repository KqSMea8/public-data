#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型预训练 微调"""
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image

kwargs={"num_classes":10} # 原本为1000类，改为10类
# model = models.densenet161(pretrained=True,**kwargs)

model = models.alexnet(pretrained=True)

from torchsummary import summary #非pytorch官方包，需额外安装
print(summary(model, (3, 224, 224)))

for p in model.parameters():
    print(p.shape)

# #https://blog.csdn.net/u012436149/article/details/78038098
# for para in list(model.parameters())[:-2]:
#     print(para)
    # para.requires_grad=False 

print("---------")
for k,v in model.state_dict().items():
    print(k,v.shape)

# model.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 10),
#         )

# print(model)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型预训练 微调"""
from __future__ import print_function
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image

kwargs={"num_classes":10} # 原本为1000类，改为10类
model = models.densenet161(pretrained=True,**kwargs)

print(model)

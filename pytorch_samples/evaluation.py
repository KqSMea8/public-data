#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image

from torchsummary import summary  # 非官方package，需额外安装


def load_img_data(img_file, img_size=224):
    """
    1. Compose的参数是list，有顺序区别，ToTensor()不能放在Normalize()之后;
    2. Normalize对于alext的预测结果的影响很大，其他的无影响？, mean,std的数值是从哪里来的？
        是因为alextnet没有nn.init.相关的操作？
    3. RandomHorizontalFlip 仅用作数据增强，对预测无影响
    4. RandomResizedCrop 裁剪图片，保证输入网络的大小一致
    5. 使用CenterCrop代替RandomResizedCrop ？
    """
    tfc = transforms.Compose([
        transforms.CenterCrop(img_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
    ])

    images = np.array([])
    image = tfc(Image.open(img_file).convert('RGB'))  # L
    images = np.append(images, image.numpy())
    print(images.shape)  # 50176*3
    # [batch_size,channels,w,h]
    img = images.reshape(-1, 3, img_size, img_size)
    data = torch.from_numpy(img).float()
    return data


def model_eval(model, img_data):
    """ 输出结果的index对应的分类名称：imagenet1000_clsid_to_human.txt
    https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    """
    model_name = type(model).__name__
    model.eval()
    with torch.no_grad():
        # img_data = img_data.to(device)
        output = model(img_data)
        pred = output.max(1, keepdim=True)[1]
        print("%s-pred:%s" % (model_name, pred[0]))


if __name__ == '__main__':
    img_file = "../data/Cat03.jpg"  # 282: 'tiger cat'
    img_data = load_img_data(img_file, 224)

    # #ll ~/.torch/models

    t = (3, 224, 224)
    model_list = ['alexnet', 'vgg16', 'resnet18',
                  'densenet161', 'squeezenet1_1']
    model_list = ['resnet18'] #
    for model_name in model_list:
        model = models.__dict__[model_name](pretrained=True)
        print(summary(model, t))  # 输出网络结构
        model_eval(model, img_data)  # 输出预测结果

    # img_data = load_img_data(img_file, 299)  # GoogLeNet的输入图片大小为299*299
    # model = models.inception_v3(pretrained=True)
    # # print(summary(model, t)) #error inception_v3, why?
    # model_eval(model, img_data)

    # # UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
    # # UserWarning: nn.init.kaiming_uniform is now deprecated in favor of nn.init.kaiming_uniform_.
    # # UserWarning: nn.init.normal is now deprecated in favor of nn.init.normal_. 

    # RuntimeError: Given input size: (2048x5x5).
    # Calculated output size: (2048x0x0).
    # Output size is too small at /Users/soumith/miniconda2/conda-bld/pytorch_1532623076075/work/aten/src/THNN/generic/SpatialAveragePooling.c:64

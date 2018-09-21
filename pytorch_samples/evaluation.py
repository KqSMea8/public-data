#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image


def load_img_data(img_file, img_size=224):
    tfc = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
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
    model_name = type(model).__name__
    model.eval()
    with torch.no_grad():
        # img_data = img_data.to(device)
        output = model(img_data)
        pred = output.max(1, keepdim=True)[1]
        print("%s-pred:%s" % (model_name, pred[0]))


# ImageNet,imagenet1000_clsid_to_human.txt
#  https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

if __name__ == '__main__':
    img_file = "../data/Cat03.jpg"  # 282: 'tiger cat'

    # #ll ~/.torch/models
    img_data = load_img_data(img_file, 224)
    model_eval(models.alexnet(pretrained=True), img_data)
    model_eval(models.vgg11(pretrained=True), img_data)
    model_eval(models.resnet18(pretrained=True), img_data)
    model_eval(models.densenet161(pretrained=True), img_data)
    model_eval(models.squeezenet1_1(pretrained=True), img_data)

    # UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
    # UserWarning: nn.init.kaiming_uniform is now deprecated in favor of nn.init.kaiming_uniform_.
    # UserWarning: nn.init.normal is now deprecated in favor of nn.init.normal_.

    # error ?
    img_data = load_img_data(img_file, 299)  # GoogLeNet的输入图片大小为299*299
    model_eval(models.inception_v3(pretrained=True), img_data)

    # RuntimeError: Given input size: (2048x5x5).
    # Calculated output size: (2048x0x0).
    # Output size is too small at /Users/soumith/miniconda2/conda-bld/pytorch_1532623076075/work/aten/src/THNN/generic/SpatialAveragePooling.c:64

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
"""
https://github.com/pytorch/examples/blob/master/mnist/main.py
网络结构有点像LeNet ？

F.max_pool2d
F.relu
F.

nn和nn.functional 的区别，需要维持状态的用nn，不需要的用functional.

# 1.卷积层 torch.nn.Conv1d,Conv2d,Conv3d 
Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) 
输入->输出计算公式？
*. in_channels(int) – 输入信号的通道
*. out_channels(int) – 卷积产生的通道
*. kerner_size(int or tuple) - 卷积核的尺寸
*. stride(int or tuple, optional) - 卷积步长
*. padding (int or tuple, optional)- 输入的每一条边补充0的层数
*. dilation(int or tuple, `optional``) – 卷积核元素之间的间距?
*. groups(int, optional) – 从输入通道到输出通道的阻塞连接数?
*. bias(bool, optional) - 如果bias=True，添加偏置


#torch.nn.ConvTranspose1d,ConvTranspose2d,ConvTranspose3d
# 解卷积操作?
ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
output_padding

# 2.池化层 torch.nn.functional.max_pool2d,max_unpool2d
max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
    *. kernel_size(int or tuple) - max pooling的窗口大小
    *. stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
    *. padding(int or tuple, optional) - 输入的每一条边补充0的层数
    *. dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数 
    *. ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作 
    *. return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)

# 3. ReLU(x)=max(0,x)
torch.nn.functional.relu(input, inplace=False) 

# 4. dropout2d
torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)

"""

'''
torch.optim.SGD mini-batch gradient descent 随机小批量

SGD(params, lr=<object object>, momentum=0, dampening=0, weight_decay=0, nesterov=False)	
    *. params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
    *. lr (float) – learning rate
    *. momentum (float, optional) – momentum factor (default: 0) 动量？
    *. weight_decay (float, optional) – weight decay (L2 penalty) (default: 0) 权重衰减 L2惩罚
    *. dampening (float, optional) – dampening for momentum (default: 0)
    *. nesterov (bool, optional) – enables Nesterov momentum (default: False)
optimizer.zero_grad()
optimizer.step()


torch.nn.functional.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean')
loss.backward()
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 10 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() #默认参数？p=0.5,inplace=False
        # 320 in_features, 50 out_features, in-out的映射关系是怎么搞的？
        self.fc1 = nn.Linear(320, 50)  #320, ?  an affine operation: y = Wx + b
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #卷积->池化->激活
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # 将多行的Tensor拼接成一行，-1的意义是让库自行计算行数或列数
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  #设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Clears the gradients of all optimized torch.Tensor s.
        optimizer.zero_grad()
        output = model(data)
        # 定义损失函数 nll_loss=The negative log likelihood loss.
        loss = F.nll_loss(output, target)
        loss.backward()  # 自动求导
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()   #设置模型为预测模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def pre():
    output = model(data)
    pred = output.max(1, keepdim=True)[1]


def data_loader(args, use_cuda):
    # load data 加载数据
    # doc，https://pytorch.org/docs/0.4.0/_modules/torchvision/datasets/mnist.html#MNIST
    tfc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    # 0.1307,0.3081 是怎么来的？
    train_datasets = datasets.MNIST('../data', train=True, download=True,
                                    transform=tfc)
    test_datasets = datasets.MNIST('../data', train=False,
                                   transform=tfc) 
    # return
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的?

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device) #设置在cpu or cuda上执行
    print(model)

    # 定义优化函数 SGD + 动量 , mini-batch gradient descent 
    # momentum 动量 
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    train_loader, test_loader = data_loader(args, use_cuda)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    
    #model save
    model_file="../data/mnist.pth"
    torch.save(model.state_dict(), model_file)
    # model = torch.load(model_file)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    main(args)

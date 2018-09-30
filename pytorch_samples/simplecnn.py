
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # nn.Module基类

        # 1 input image channel, 10 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # 默认参数？p=0.5,inplace=False
        # 320 in_features, 50 out_features, in-out的映射关系是怎么搞的？
        # 320, ?  an affine operation: y = Wx + b
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # return self.forward_debug(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 卷积->池化->激活
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将多行的Tensor拼接成一行，-1的意义是让库自行计算行数或列数
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def forward_debug(self, x):
        # 一步步的打印太麻烦了，是否可以循环逐层打印data.shape?
        print('x.data.shape:', x.data.shape)  # [1, 1, 28, 28]

        # [1, 10, 24, 24] , 28->24 ,
        # 卷积输出大小计算公式: N = (W − F + 2P )/S+1
        # 24 = (28 - 5 + 2*0 )/1 + 1 w=28,f=5,p=0,s=1
        print('conv1.data.shape:', self.conv1(x).data.shape)

        # [1, 10, 12, 12] 24->12 池化层输出大小计算公式?
        # max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
        # stride Default value is kernel_size  12 = (24-2+2*0)/2+1
        # floor((L_{in} + 2padding - dilation(kernel_size - 1) - 1)/stride + 1
        # (24 + 2*0 - 1*(2-1) - 1)/2 + 1
        print('max-pool2d-conv1.data.shape:',
              F.max_pool2d(self.conv1(x), 2).data.shape)  #

        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 卷积->池化->激活

        # [1, 10, 12, 12] relu 不改变大小
        print('relu-max_pool2d-conv1.data.shape:',
              x.data.shape)
        print('conv2.data.shape:', self.conv2(x).data.shape)  # [1, 20, 8, 8]

        #[1, 20, 8, 8], conv2_drop不改变大小
        print('conv2_drop-conv2.data.shape:',
              self.conv2_drop(self.conv2(x)).data.shape)

        print('max_pool2d-conv2_drop-conv2.data.shape:',
              F.max_pool2d(self.conv2_drop(self.conv2(x)), 2).data.shape)  # [1, 20, 4, 4]

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        print('x:', x.data.shape)  # [1, 20, 4, 4]

        # 320是从哪里来的？ 20*4*4 宽*高*通道数
        x = x.view(-1, 320)  # 将多行的Tensor拼接成一行，-1的意义是让库自行计算行数或列数

        print('x:', x.data.shape)  # [1, 320]
        x = F.relu(self.fc1(x))
        print('x:', x.data.shape)
        x = F.dropout(x, training=self.training)
        print('x:', x.data.shape)
        x = self.fc2(x)
        print('x:', x.data.shape)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = SimpleCNN()
    print(model)

    # https://github.com/sksq96/pytorch-summary
    from torchsummary import summary #非pytorch官方包，需额外安装
    print(summary(model, (1, 28, 28)))

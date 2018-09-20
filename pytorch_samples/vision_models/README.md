pytorch官方模型
===
官方wiki
https://pytorch.org/docs/0.4.0/torchvision/models.html?highlight=torchvision%20models

中文wiki
https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-models/

源码：
https://github.com/pytorch/vision/tree/master/torchvision/models

其他文档：
PyTorch预训练：
https://zhuanlan.zhihu.com/p/25980324 
torchvision.models源码解读：
https://blog.csdn.net/u014380165/article/details/79119664


load_state_dict方法还有一个重要的参数是strict，该参数默认是True，表示预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）。

AlexNet
    features
    classifier 这两个分离？
VGG
ResNet
    在ResNet网络结构的构建中有很多重复的子结构，这些子结构就是通过Bottleneck类来构建的。
SqueezeNet
DenseNet

可以直接使用



PyTorch使用及源码解读
https://blog.csdn.net/column/details/19413.html

https://blog.csdn.net/u014380165/article/details/79222243
1. 定义模型
    可以自己定义，可以导入torchvision.models中官方定义的模型
2. 训练、测试
    定义优化函数、损失函数
3. 数据导入准备
python的PIL库读进来的图像内容，输入对象都是PIL Image
torchvision.datasets.ImageFolder只是返回list
torch.utils.data.DataLoader类可以将list类型的输入数据封装成Tensor数据格式，以备模型使用
4. 训练

训练一个图像分类模型
https://blog.csdn.net/u014380165/article/details/78525273 






IoU,交并比，>=0.5，可接受
非最大值抑制 No-max suppression
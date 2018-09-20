pytorch入门资料
==
PyTorch深度学习：60分钟入门
https://zhuanlan.zhihu.com/p/25572330

https://pytorch.org/tutorials/
https://pytorch.org/docs/stable/index.html

https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

Adam：
https://juejin.im/entry/5983115f6fb9a03c50227fd4
2015 年 ICLR 论文（Adam: A Method for Stochastic Optimization）
优势：
    直截了当地实现
    高效的计算
    所需内存少
    梯度对角缩放的不变性
    适合解决含大规模数据和参数的优化问题
    适用于非稳态（non-stationary）目标
    适用于解决包含很高噪声或稀疏梯度的问题
    超参数可以很直观地解释，并且基本上只需极少量的调参
Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。
适应性梯度算法（AdaGrad），为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。
均方根传播（RMSProp），基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。

数据增强的工具？


https://www.jianshu.com/p/b5e93fa01385
dropout,防止过拟合提高效果, 是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。


优化函数：torch.optim
    SGD，小批量随机梯度
    Adam，adaptive moment estimation，适应性矩估计，非凸优化问题，随机梯度下降算法的扩展式，通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。 一阶优化算法？
    AdaGrad，RMSProp ？ 
    LBFGS，？


损失函数：torch.nn.functional.   
https://blog.csdn.net/zhangxb35/article/details/72464152
https://zhuanlan.zhihu.com/p/32626442 
    l1_loss
    mse_loss #L2？
    binary_cross_entropy_with_logits
    binary_cross_entropy
    cross_entropy
    cosine_embedding_loss 
    hinge_embedding_loss 
    margin_ranking_loss 
    multi_margin_loss 
    multilabel_margin_loss 
    multilabel_soft_margin_loss 
    nll_loss 
    poisson_nll_loss 
    smooth_l1_loss 
    soft_margin_loss 
    triplet_margin_loss 
    kl_div


https://github.com/yunjey/pytorch-tutorial


图像的话，可以用Pillow, OpenCV。 torchvision包
声音处理可以用scipy和librosa。
文本的处理使用原生Python或者Cython以及NLTK和SpaCy都可以。

https://github.com/chenyuntc/pytorch-book

https://www.zhihu.com/question/55720139
https://github.com/ritchieng/the-incredible-pytorch


https://github.com/pytorch/examples
1.mnist,
2.mnist_hogwild,
3.imagenet


https://github.com/ayooshkathuria/pytorch-yolo-v3
卷积神经网络的工作原理，包括残差块、跳过连接和上采样；
目标检测、边界框回归、IoU 和非极大值抑制（NMS）；
基础的 PyTorch 使用，会创建简单的神经网络；
阅读 YOLO 三篇论文，了解 YOLO 的工作原理。



用深度学习解决计算机视觉问题
1. Andrew Ng 
https://mooc.study.163.com/learn/2001281004?tid=2001392030#/learn/announce

2. ReLu vs Sigmoid ?
https://blog.csdn.net/algorithm_image/article/details/78042429
深层网络sigmoid计算两大
ReLu更接近生物神经元
sigmoid 两端饱和，在传播过程中容易丢弃信息
容易就会出现梯度消失

avg_pool vs max_pool

pytorch代码实现这些经典的网络？


图像：
https://zhuanlan.zhihu.com/p/30504700

计算机视觉的一些基础知识

1998 LeNet 
    输入: 32*32*1
2006 Hilton BP反向传播
2012 AlexNet 
    输入: 227*227*3
    ReLu、Dropout、最大池化、LRN（Local Response Normalization，局部响应归一化）、GPU加速
2014 GoogleNet Inception Module 反复堆叠高效的卷积网络结构 19层
2014 VGG 拓展性强，泛化性比较好，可用作迁移学习
2015 ResNet 残差学习 152层, 
    解决网络结构太深导致梯度消失或梯度爆炸，跳远连接


局部连接、权值共享
一个卷积层可以有多个不同的卷积核，而每一个卷积核都对应一个滤波后映射出的新图像，同一个新图像中每一个像素都来自完全相同的卷积核，就就是卷积核的权值共享。
池化层中降采样
卷积层，池化层，全连接层，Softmax层

池化层，将分辨率较高的图片转化为分辨率较低的图片。
过滤器（filter）= 卷积核（kernel）
即便节点矩阵是三维的，卷积核的尺寸只需指定两个维度。一般地，卷积核的尺寸是3×3和5×5？


1.图像分类
2.目标检测
https://blog.csdn.net/ice_actor/article/details/78574612

https://www.cnblogs.com/skyfsm/p/6806246.html
RCNN->SppNET->Fast-RCNN->Faster-RCNN
预测出（x,y,w,h）四个参数的值。

https://blog.csdn.net/u011746554/article/details/74999010
Faster RCNN：将特征抽取(feature extraction)，proposal提取，bounding box regression(rect refine)，classification都整合在了一个网络中，使得综合性能有较大提高，在检测速度方面尤为明显。

https://www.leiphone.com/news/201805/PwgtroHpKQL7SrO2.html
用区域提议方法（region proposal method）生成感兴趣区域 ( regins of interest, ROIs ) 来进行目标检测
选择性搜索算法（Selective Search, SS）
区域提议网络(Region proposal network)

https://www.jianshu.com/p/cef69c6651a9
R-CNN：Region-based Convolutional Neural Networks
normalized initialization and BN(Batch Normalization)的提出，解决了梯度消失或爆炸问题

候选区域（Region Proposal）
-Selective Search
-Edge Boxes
一是最后一个卷积层后加了一个ROI pooling layer，
二是损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练。
端到端训练测试？

使用CNN直接产生Region Proposal并对其分类？

RPN(Region Proposal Networks)网络，
滑动窗口
Anchor机制，边框回归
多尺度多长宽比的Region Proposal
Anchor是滑动窗口的中心，它与尺度和长宽比相关
将一直以来分离的region proposal和CNN分类融合到了一起，使用端到端的网络进行目标检测，无论在速度上还是精度上都得到了不错的提高。
还是达不到实时的目标检测。
YOLO这类目标检测方法的出现让实时性也变的成为可能。

R-FCN(基于区域的检测器）

https://zhuanlan.zhihu.com/p/25045711
只需瞄一眼（You Only Look Once，YOLO）
deformable parts models (DPM)
post-processing来去除重复bounding boxes来进行优化
YOLO采用单个卷积神经网络来预测多个bounding boxes和类别概率
Fast R-CNN检测方法会错误的将背景中的斑块检测为目标，原因在于Fast R-CNN在检测中无法看到全局图像
相对于Fast R-CNN，YOLO背景预测错误率低一半。


进击的YOLOv3，目标检测网络的巅峰之作 | 内附实景大片
https://www.jiqizhixin.com/articles/2018-05-14-4


https://cloud.tencent.com/developer/article/1086374


https://github.com/pytorch/examples


针对 Intel 架构调节了 MKL

反向模式自动微分
Torch 和 PyTorch 共享相同的后端代码？

AlexNet、VGG 和 ResNet。



nn.conv2d 过程验证
https://zhuanlan.zhihu.com/p/32190799

单通道 - 灰度？
多通道 - rgba？


图像概念梳理：
https://blog.csdn.net/futurewu/article/details/9945611


图像分割算法？

人脸验证，人脸识别

similarity 

siamese network
facenet 
https://zhuanlan.zhihu.com/p/35040994
三元组损失函数？triplet  positive(正例) - anchor(锚) - negative(反例)  APN

pytorch 中国车牌识别？

动画演示：
http://imgtec.eetrend.com/blog/9715
https://github.com/vdumoulin/conv_arithmetic

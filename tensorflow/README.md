问题：
1. Tensorflow实现分布式计算的机制？

1. install 
pip install tensorflow --user -U
pip install tensorboard --user -U

TensorFlow建议使用二进制的TFRecords格式，这样可以支持QueuRunner和Coordinator进行多线程数据读取，并且可以通过batch size和epoch参数来控制训练时单次batch的大小和对样本文件迭代训练多少轮。

CSV与TFRecords格式转换工具convert_cancer_to_tfrecords.py

python-gflags项目

Sgd、Rmsprop还是选择Adagrad、Ftrl

TensorBoard
Protobuf

Google Cloud ML服务

分布式TensorFlow中ps、worker、in-graph、between-graph、synchronous training和asynchronous training的概念。
* ps是整个训练集群的参数服务器，保存模型的Variable
* worker是计算模型梯度的节点，得到的梯度向量会交付给ps更新模型
* in-graph与between-graph对应，但两者都可以实现同步训练和异步训练，
* in-graph指整个集群由一个client来构建graph，并且由这个client来提交graph到集群中，其他worker只负责处理梯度计算的任务，
* between-graph指的是一个集群中多个worker可以创建多个graph，但由于worker运行的代码相同因此构建的graph也相同，并且参数都保存到相同的ps中保证训练同一个模型，这样多个worker都可以构建graph和读取训练数据，适合大数据场景。
* synchronous training 同步训练每次更新梯度需要阻塞等待所有worker的结果
* asynchronous training 异步训练不会有阻塞，训练的效率更高，在大数据和分布式的场景下一般使用异步训练。

应用：
1. 抓取网页内容 根据已有数据，自动解析网页内容？
2. 为每个feed建知识图谱？提取实体、属性，建立实体关系；
3. 公共知识图谱库
4. 个人知识图谱 - 仿人脑的记忆功能？

WARNING:tensorflow:From /Users/baidu/Library/Python/2.7/lib/python/site-packages/tensorflow/python/util/tf_should_use.py:170: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
2017-08-08 17:21:47.545545: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-08 17:21:47.545572: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-08 17:21:47.545577: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-08 17:21:47.545581: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.

http://www.52cs.org/?p=1157
https://github.com/tobegit3hub/tensorflow_examples

http://blog.csdn.net/lenbow/article/details/52152766

代价模型

通讯
RDMA？
心跳检测
自动求导

SWIG?

evealuate
grid search
cross validation

spark 大规模数据
tensorflow 数据规模内存可加载

TensorBoard 安装？
beam search ?
必须构建静态图
CPU上的矩阵运算库Eigen? BLAS?

loss function?

cross-entropy

cuDNN?

keras, 可运行在Tensorflow上

前馈网络

python anaconda

softmax regression,多分类任务

无监督自编码(autoEncoder)
SGD参数？

Dropout - 减轻过拟合,
Adagrad - 自适应学习速率,
ReLU - 解决梯度弥散的激活函数

bagging

梯度弥散(Gradient vanishment)
激活函数：Sigmoid,ReLU,Softplus
ReLu变种：EIU,PReLU
XOR问题

卷积神经网络,Convolutional Neural Network

SIFT,HoG
卷积的权值共享
抽取特征-核滤波，抗行变 - 激活函数，最大池化max-pooling
全连接，局部连接

数据增强(data augmentation)

AlexNet,VGGnet,Google Inception Net,ResNet
LRN

~~
循环神经网络 RNN

稠密向量
向量表达
最大似然

强化学习,连续决策 (Reinforcement Learning)
* Environment State
* Action
* Reward 根据时间衰减系数
有序列输出


deepmind
    策略网络 Policy-network
    估值网络 value-network,即DQN
    蒙特卡洛树 Monte Carlo Tree search
    快速走子Fast Rollout

Policy Gradients
    Policy-Based
    Value-Based, 所有action的Q值，取最高的

OpenAI-Gym, 强化学习的环境生成工具，标准化环境，对算法对比测试
可以大批量深度学习
目标变化的、不明确的
DQN,DeepQ-network

未来Reward衰减系数


自我意识
    什么是自我意识？
    高级自我意识， 镜子实验，只有人类(4岁以上)，黑猩猩、大象、海豚等动物才能识别出镜子里出现的是自己？

策略网络，根据环境制定行动

Q-Learning

有限马尔可夫决策过程

TF.Learn
Estimator 
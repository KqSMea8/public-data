问题：
1. Tensorflow实现分布式计算的机制？

1. install 
pip install tensorflow --user -U

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

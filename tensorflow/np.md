179	np.array	构建numpy数组	
21	np.random.normal	正态分布
2	np.random.rand	 随机样本位于[0, 1)中
1	np.random.randn	 从标准正态分布中返回一个或多个样本值
1	np.random.randint (0,n] 正整数	
24	np.random.seed	如果使用相同的seed()值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
4	np.random.uniform	从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
4	np.random.permutation	返回一个洗牌后的矩阵副本
2	np.random.shuffle 对当前矩阵洗牌
104	np.random.choice 随机选取
10	np.repeat 对数组中的元素进行连续重复复制	
9	np.arange 根据start与stop指定的范围以及step设定的步长，生成一个 ndarray	
0   np.range 根据start与stop指定的范围以及step设定的步长，生成一个序列
9	np.zeros	返回一个给定形状和类型的用0填充的数组
1	np.zeros_like	
3	np.ones	单位矩阵
0	np.ones_like	单位矩阵
18	np.argmax	最大数的索引
4	np.array_split 将narray分成几份	
1	np.c_	将切片对象沿第二个轴（按列）转换为连接
3	np.squeeze	从数组的形状中删除单维条目，即把shape中为1的维度去掉
6	np.ceil	向正无穷取整朝正无穷大方向取整
27	np.round	
4	np.column_stack	 列合并/扩展
8	np.concatenate 数组拼接	
6	np.expand_dims	axis的那一个轴上把数据加上去
1	np.abs	绝对值
13	np.mean	 均值
7	np.sum	求和
2	np.min	最小值
1	np.float32
1	np.uint8
1	np.int32
4	np.hstack	horizontal-stack	
14	np.linspace	在指定的间隔内返回均匀间隔的数字
3	np.matmul	矩阵相乘
8	np.meshgrid	从坐标向量返回坐标矩阵
16	np.nan_to_num	nan-inf
7	np.reshape	
4	np.resize	
9	np.roll	 将a，沿着axis的方向，滚动shift长度
2	np.setdiff1d	返回一个数组, 其中包含在第一个输入数组而不在第二个输入数组的元素
1	np.sin	 sin()
4	np.sort	排序
10	np.sqrt	平方根
114	np.transpose 行交换？
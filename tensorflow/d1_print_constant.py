#!/usr/bin/python
# -*- coding: utf-8 -*-
#https://github.com/tobegit3hub/tensorflow_examples

import tensorflow as tf

#避免Warn输出 The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #

print("Run TensorFlow to multiple numbers")

# import pudb;pudb.set_trace() #pudb python调试工具
#http://python.jobbole.com/82638/

#http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html

number = tf.constant(3)
#Variable

with tf.Session() as sess:
  print(sess.run(number))

print("End of TensorFlow")

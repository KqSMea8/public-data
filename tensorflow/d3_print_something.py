#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf

print("Run TensorFlow to multiple numbers")

a = tf.placeholder("float")
b = tf.placeholder("float")
# c = tf.mul(a, b) #'module' object has no attribute 'mul'
c= tf.multiply(a,b) #版本升级 mul -> multiply
# tf.mul, tf.sub and tf.neg are deprecated 
# tf.multiply, tf.subtract and tf.negative.

# feed_dict?  feed机制, 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁
# placeholder() 创建占位符

with tf.Session() as sess:
    print("Multiple 1 with 2 to get:")
    print(sess.run(c, feed_dict={a: 1, b: 2}))  # Assert 2

    print("Multiple 3 with 4 to get:")
    print(sess.run(c, feed_dict={a: 3, b: 4}))  # Assert 12

print("End of TensorFlow")

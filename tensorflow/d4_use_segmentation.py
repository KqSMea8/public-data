#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf

data = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
result = tf.segment_sum(data, tf.constant([0, 0, 1]))
#segment_sum？

with tf.Session() as sess:
    print(sess.run(result))

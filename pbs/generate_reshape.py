#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:27:12 2019

@author: vgerous
"""
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.constant(
	[
	[
	[[1, 2], [3, 4]],
	[[5, 6], [7, 8]]
	],
	[
	[[9, 10], [11, 12]],
	[[13, 14], [15, 16]]
	]
	]
	)
    op = tf.reshape(x, [1,2,2,4], name='op_to_store_reshape')
    print(x)
    print(op)
    sess.run(tf.global_variables_initializer())
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store_reshape'])
    with tf.gfile.FastGFile(pb_file_path+'/model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
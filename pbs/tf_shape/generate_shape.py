#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import imageio
in_img = imageio.imread('in.bmp')
in_img = in_img.astype(np.float32)/255.0
in_data = in_img[np.newaxis, :]
x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='dnn_in')
y = tf.shape(x, name='dnn_out')
sess=tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run(y, feed_dict={x: in_data})
graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'shape.pb', as_text=False)
print(sess.run(y, feed_dict={x: in_data}))
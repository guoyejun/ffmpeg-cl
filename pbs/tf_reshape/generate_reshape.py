#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:27:12 2019

@author: vgerous
"""
import tensorflow as tf
import numpy as np
import imageio
in_img = imageio.imread('in.bmp')
in_img = in_img.astype(np.float32)/255.0
in_data = in_img[np.newaxis, :]
x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='dnn_in')
assert(in_img.shape[0]*in_img.shape[1]%250 == 0)
y = tf.reshape(x, [1,-1,250,3], name='dnn_out')

sess=tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run(y, feed_dict={x: in_data})
graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
tf.train.write_graph(graph_def, '.', 'reshape.pb', as_text=False)
print('in_data.shape: ', in_data.shape)
print('output.shape: ', output.shape)

output = output * 255.0
output = output.astype(np.uint8)
imageio.imsave("out.bmp", np.squeeze(output))

tf.reset_default_graph()
with open('reshape.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")
    tf.summary.FileWriter('/tmp/graph', tf.get_default_graph())
    print('run tensorboard --logdir /tmp/graph')
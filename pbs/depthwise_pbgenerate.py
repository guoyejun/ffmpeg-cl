import tensorflow as tf
import numpy as np
import scipy.misc

def complex_depthwiseconv2dnative_pb_generator():
    '''
    When function depthwise_conv2d has a dilation>1, graph is composed of ~50 nodes
    '''
    in_img = scipy.misc.imread('in.bmp')
    in_img = in_img.astype(np.float32)/255.0
    in_data = in_img[np.newaxis, :]
    filter_data = np.array([0.5, 0, 0]).reshape(1,1,3,1).astype(np.float32)
    filter = tf.Variable(filter_data)
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='dnn_in')
    y = tf.nn.depthwise_conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME', name='dnn_out',rate=(2,2))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(y, feed_dict={x: in_data})
    graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
    tf.train.write_graph(graph_def, '.', 'complex_depthwise_conv2d.pb', as_text=False)
    output = output * 255.0
    output = output.astype(np.uint8)
    scipy.misc.imsave("out.bmp", np.squeeze(output))
    
def simple_depthwiseconv2dnative_pb_generator():
    in_img = scipy.misc.imread('in.bmp')
    in_img = in_img.astype(np.float32)/255.0
    in_data = in_img[np.newaxis, :]
    filter_data = np.array([0.5, 0, 0]).reshape(1,1,3,1).astype(np.float32)
    filter = tf.Variable(filter_data)
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='dnn_in')
    y = tf.nn.depthwise_conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME', name='dnn_out')
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(y, feed_dict={x: in_data})
    graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dnn_out'])
    tf.train.write_graph(graph_def, '.', 'simple_depthwise_conv2d.pb', as_text=False)
    output = output * 255.0
    output = output.astype(np.uint8)
    scipy.misc.imsave("out.bmp", np.squeeze(output))

##complex_depthwiseconv2dnative_pb_generator()
simple_depthwiseconv2dnative_pb_generator()

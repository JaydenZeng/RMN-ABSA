import numpy as np
import tensorflow as tf
from Settings import Config


def conv(inputs, adj, W):
    shape_1 = tf.shape(inputs)[0]
    shape_2 = tf.shape(inputs)[1]
    shape_3 = tf.shape(inputs)[2]
    adj = tf.cast(adj, tf.float32)
    hid = tf.reshape(tf.matmul(tf.reshape(inputs, [-1,shape_3]), W), [shape_1, shape_2, -1])
    denom = tf.reduce_sum(adj, -1, keep_dims=True) +1
    output = tf.nn.relu(tf.matmul(adj, hid)/denom)
    return output

def conv_relu(inputs, adj, W, b):
    shape_1 = tf.shape(inputs)[0]
    shape_2 = tf.shape(inputs)[1]
    shape_3 = tf.shape(inputs)[2]
    adj = tf.cast(adj, tf.float32)
    hid = tf.reshape(tf.matmul(tf.reshape(inputs, [-1,shape_3]), W)+b, [shape_1, shape_2, -1])
    output = tf.nn.relu(tf.matmul(adj, hid))
    return output

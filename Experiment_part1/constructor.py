# -*- coding: utf-8 -*-
import tensorflow as tf

def inference(x):

    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('D_0_w', [x.get_shape()[1], 1024], initializer=w_init)
    b0 = tf.get_variable('D_0_b', [1024], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # 2nd hidden layer
    w1 = tf.get_variable('D_1_w', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.get_variable('D_1_b', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.get_variable('D_2_w', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.get_variable('D_2_b', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output layer
    w3 = tf.get_variable('D_3_w', [h2.get_shape()[1], 10], initializer=w_init)
    b3 = tf.get_variable('D_3_b', [10], initializer=b_init)
    o1 = tf.matmul(h2, w3) + b3
    o = tf.nn.relu(o1)
    return o1
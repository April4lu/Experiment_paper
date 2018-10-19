# -*- coding: utf-8 -*-
import tensorflow as tf

def cross_entropy(y,y_):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels= tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    return cross_entropy_mean
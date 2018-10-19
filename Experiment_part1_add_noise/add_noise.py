# -*- coding: utf-8 -*-
import tensorflow as tf

def AddGaussianNoise(t,sigma):
    noisy_t = t + tf.random_normal(tf.shape(t),stddev=sigma)
    return noisy_t
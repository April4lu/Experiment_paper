# -*- coding: utf-8 -*-
import tensorflow as tf
from add_noise import AddGaussianNoise

def DpGradientDescentOptimizer_noise(learning_rate,loss,var_list,global_step,sigma):
    grad_SSEs_list = []
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.name_scope("gradients"):
        grad_var_list = optimizer.compute_gradients(loss=loss, var_list=var_list)
        
    with tf.name_scope("readgradients"):
        grad_list = [grad for (grad, var) in grad_var_list]
#        grad_SSES = [tf.reduce_sum(var) for var in grad_list]
#        grad_SSES1 = [tf.reduce_sum(var**2) for var in grad_list]
#        tf.summary.histogram("grad_SSES",grad_SSES)
#        tf.summary.histogram("grad_SSES1",grad_SSES1)
        
        grad_list_noise = [AddGaussianNoise(t,sigma) for t in grad_list]
#        grad_SSES_noise = [tf.reduce_sum(param) for param in grad_list_noise]
        grad_SSEs_list.extend(grad_list_noise)   
#        tf.summary.histogram("grad_SSES_noise",grad_SSES_noise)
        
        grad_var_list_noise = [(grad_SSEs_list[i], grad_var_list[i][1]) for i in range(len(grad_list_noise))]
        
    with tf.name_scope("updategradients"):
        train_op = optimizer.apply_gradients(grad_var_list_noise, global_step=global_step)
        
    return train_op
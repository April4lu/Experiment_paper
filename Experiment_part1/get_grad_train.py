# -*- coding: utf-8 -*-
import tensorflow as tf

def DpGradientDescentOptimizer(learning_rate,loss,var_list,global_step):
    #grad_SSEs_list = []
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.name_scope("gradients"):
        grad_var_list = optimizer.compute_gradients(loss=loss, var_list=var_list)
#    with tf.name_scope("readgradients"):
#        grad_list = [grad for (grad, var) in grad_var_list]
#        #var_list = [grad for (grad, var) in grad_var_list]
#        grad_SSES = [tf.reduce_sum(var**2) for var in grad_list]
#        grad_SSEs_list.extend(grad_list)
#        tf.summary.histogram("grad_SSES",grad_SSES)
        
    with tf.name_scope("updategradients"):
        train_op = optimizer.apply_gradients(grad_var_list, global_step=global_step)  
    return train_op
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
#from tensorflow.examples.tutorials.mnist import input_data 


def generator(inputs_g):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('G_0_w', [inputs_g.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('G_0_b', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(inputs_g, w0) + b0)

    # 2nd hidden layer
    w1 = tf.get_variable('G_1_w', [h0.get_shape()[1], 512], initializer=w_init)
    b1 = tf.get_variable('G_1_b', [512], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.get_variable('G_2_w', [h1.get_shape()[1], 1024], initializer=w_init)
    b2 = tf.get_variable('G_2_b', [1024], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.get_variable('G_3_w', [h2.get_shape()[1], 784], initializer=w_init)
    b3 = tf.get_variable('G_3_b', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h2, w3) + b3)

    return o

# D(x)
def discriminator(inputs_d, drop_out):
    reader = pywrap_tensorflow.NewCheckpointReader((r"./nn/train_model.ckpt1"))
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)
    # 1st hidden layer
    w0 = tf.get_variable('D_0_w', initializer=reader.get_tensor('D_0_w'),trainable=False)
    b0 = tf.get_variable('D_0_b', initializer=reader.get_tensor('D_0_b'),trainable=False)
    h0 = tf.nn.relu(tf.matmul(inputs_d, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)
    # 2nd hidden layer
    w1 = tf.get_variable('D_1_w', initializer=reader.get_tensor('D_1_w'),trainable=False)
    b1 = tf.get_variable('D_1_b', initializer=reader.get_tensor('D_1_b'),trainable=False)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)
    # 3rd hidden layer
    w2 = tf.get_variable('D_2_w', initializer=reader.get_tensor('D_2_w'),trainable=False)
    b2 = tf.get_variable('D_2_b', initializer=reader.get_tensor('D_2_b'),trainable=False)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)
    # output layer     
    w3 = tf.get_variable('D_3_w', [h2.get_shape()[1], 1], initializer=w_init,trainable=True)
    b3 = tf.get_variable('D_3_b', [1], initializer=b_init,trainable=True)
    o = tf.sigmoid(tf.matmul(h2, w3) + b3)
    return o


batch_size = 100
learning_rate = 0.0001
train_epoch = 4#由于只检查权重的不更新，所以迭代步数设置较少
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#train_set = (mnist.train.images - 0.5) / 0.5

with tf.variable_scope('G'):
    inputs_g = tf.placeholder(tf.float32, shape=(None, 100))
    G_z = generator(inputs_g)
    
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    #inputs_d = tf.placeholder(tf.float32, shape=(None, 784))
    #D_real = discriminator(inputs_d,drop_out,t_vars)
    #scope.reuse_variables()
    
    D_fake= discriminator(G_z,drop_out)


eps = 1e-2    
#D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
D_loss = tf.reduce_mean(- tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network

t_vars = tf.trainable_variables()
print(t_vars)
D_vars = [var for var in t_vars if 'D_3_' in var.name]
print(D_vars)
G_vars = [var for var in t_vars if 'G_' in var.name]
#a_vars = tf.all_variables()
#print(a_vars)

save_file = './cc3/train_model.ckpt1'
saver = tf.train.Saver()

# optimizer for each network
D_optim = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)


with tf.Session()as sess:
    
    sess.run(tf.global_variables_initializer())
    #训练开始
    #G_losses = []
    #D_losses = []
    for epoch in range(train_epoch):
        print(epoch)
        G_losses = []
        D_losses = []
        
        #if epoch%10 == 0:
            #print("G_losses:",G_losses)
            #print("D_losses:",D_losses)

        for iter in range(55000 // batch_size):
        #for iter in range(train_set.shape[0] // batch_size):
            
            #x_ = train_set[iter*batch_size:(iter+1)*batch_size]
            z_ = np.random.normal(0, 1, (batch_size, 100))
            #loss_d_,_ = sess.run([D_loss, D_optim], feed_dict={inputs_d:x_, inputs_g:z_, drop_out: 0.3})
            loss_d_, _ = sess.run([D_loss, D_optim], {inputs_g: z_, drop_out: 0.3})
            D_losses.append(loss_d_)
            
            z_ = np.random.normal(0, 1, (batch_size, 100))
            loss_g_, _ = sess.run([G_loss, G_optim], {inputs_g: z_, drop_out: 0.3})
            G_losses.append(loss_g_)
        #if epoch%10 == 0:
            #print("G_losses:",G_losses)
            #print("D_losses:",D_losses)
    saver.save(sess, save_file)

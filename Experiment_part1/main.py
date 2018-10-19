# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data 
from get_grad_train import DpGradientDescentOptimizer
from constructor import inference
from compute_cross_entropy import cross_entropy
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

input_units = 784
output_units = 10
batch_size = 100
learning_rate = 0.8
training_steps = 200


def train(mnist):
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, input_units],name='x_input')
        y_ = tf.placeholder(tf.float32,[None,output_units],name = 'label')
    
    with tf.name_scope('Name'):
        y = inference(x)
        global_step = tf.Variable(0, trainable=False)
        
    with tf.name_scope('Xent'):        
        cross_entropy_mean = cross_entropy(y,y_)
        
    with tf.name_scope('Loss'):
        #计算总的损失
        loss = cross_entropy_mean 
    with tf.name_scope('Optimizer'):
        
        t_vars = tf.trainable_variables()
        N_vars = [var for var in t_vars]
        train_op = DpGradientDescentOptimizer(learning_rate,loss,N_vars,global_step)
        
    with tf.name_scope('Accuracy'):
        #定义准确率
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    save_file = './nn/train_model.ckpt1'
    saver = tf.train.Saver()    
    #初始化会话
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        validate_data = {x:mnist.validation.images,y_:mnist.validation.labels}        
        test_data = {x:mnist.test.images,y_:mnist.test.labels}
        #writer = tf.summary.FileWriter('D:/log_change',sess.graph)
        #merged = tf.summary.merge_all()
        #训练开始
        for i in range(training_steps):
            
            if i%10 == 0:
                validation_acc = sess.run(accuracy,feed_dict = validate_data)
                print("After %d training steps, validation accuracy using average model is %g"%(i, validation_acc))
            
            for batch_i in range(mnist.train.num_examples//batch_size):
                xs,ys = mnist.train.next_batch(batch_size)
                sess.run(train_op,feed_dict={x:xs,y_:ys})
                
                #summary = sess.run(merged,feed_dict={x:xs,y_:ys})           
                #writer.add_summary(summary,batch_i)
            
        test_acc = sess.run(accuracy,feed_dict = test_data)
        print("After %s training steps, test accuracy using average model is %g"%(training_steps, test_acc))
        saver.save(sess, save_file)
    #writer.close
def main(argv=None):
    train(mnist)
if __name__ =='__main__':
    tf.app.run()

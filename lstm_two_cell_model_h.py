# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np
import rnn_static_rnn_minus
import rnn_BasicLSTMCell_minus2

def transfer_labels(labels):

	indexes = np.unique(labels)

	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]

	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label
	return labels, num_classes
 
def load_data(data, label, batch_size):
    while 1:
        batch_num = int(math.ceil(data.shape[0] / batch_size))
        for i in range(batch_num):
            if i != batch_num - 1:
                data_batch = data[i * batch_size: (i + 1) * batch_size]
                label_batch = label[i * batch_size: (i + 1) * batch_size]
            else:
                data_batch = data[i * batch_size:]
                label_batch = label[i * batch_size:]
            yield [data_batch, label_batch]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def time_minus(data_matrix):

    temp_zeros = np.zeros((data_matrix.shape[0], 1, data_matrix.shape[2]))
    data_matrix_temp = np.column_stack((temp_zeros, data_matrix))
    data_matrix_temp = np.delete(data_matrix_temp, -1, axis=1)
    data_matrix_temp = data_matrix - data_matrix_temp
    data_matrix_temp = np.delete(data_matrix_temp, 0, axis=1)
    return np.column_stack((temp_zeros, data_matrix_temp))
    
class model(object):
    def __init__(self, data, num_class, parameters):
        self.batch_size = None
        self.n_steps = data.shape[1]
        self.input_dim = data.shape[2]
        self.num_class = num_class
        
        self.filter_height_size = parameters["filter_height_size"]    #   [2, 3, 5]
        self.filter_nums = parameters["filter_nums"]    #   128
        self.learning_rate = parameters["learning_rate"] #1e-3
        self.lstm_hidden_size = parameters["lstm_hidden_size"]
        
        self.on_train = tf.placeholder(tf.bool,[],"whether_train")
        global_step = tf.Variable(0, trainable=False)
        
        with tf.name_scope("input"):
            self.droup_keep_prob = tf.placeholder(tf.float32,[],"droup_keep_prob")
            self.x_keep_prob = tf.placeholder(tf.float32,[],"x_keep_prob")
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.input_dim], "input")
            self.x_drop = tf.nn.dropout(self.x, self.x_keep_prob)
            self.x_minus = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.input_dim], "input2")
            self.x_minus_drop = tf.nn.dropout(self.x_minus, self.x_keep_prob)
            self.y = tf.placeholder(tf.int32, [self.batch_size], 'labels')
            self.y_one_hot = tf.one_hot(self.y, self.num_class)
                
        with tf.name_scope('lstm'):
            self.lstm_input = tf.unstack(self.x_drop, axis = 1)
            self.lstm_input_minus = tf.unstack(self.x_minus_drop, axis = 1)
            lstm_cell = rnn_BasicLSTMCell_minus2.BasicLSTMCell(self.lstm_hidden_size, forget_bias=1.0)
            self.lstm_outputs, states, states_minus = rnn_static_rnn_minus.static_rnn(lstm_cell, self.lstm_input, self.lstm_input_minus, dtype=tf.float32)
            self.lstm_outputs_stack = tf.stack(self.lstm_outputs, axis = 1)
            self.lstm_outputs_dp = tf.nn.dropout(self.lstm_outputs_stack, self.droup_keep_prob)
            print self.lstm_outputs_dp.shape
            self.lstm_final = self.lstm_outputs_dp
            
        with tf.name_scope("cnn1"):
            self.W_cnn = weight_variable([self.filter_height_size[0], self.lstm_hidden_size*2, 1, self.filter_nums])
            self.B_cnn = bias_variable([self.filter_nums])
            self.conv1 = tf.nn.relu(tf.nn.conv2d(tf.expand_dims(self.lstm_final, axis=-1), self.W_cnn, strides = [1,1,self.lstm_hidden_size*2,1], padding='SAME') + self.B_cnn)    ##(?, 40, 1, 512)
            self.conv1_dp = tf.nn.dropout(self.conv1, self.droup_keep_prob)
            
        with tf.name_scope("cnn2"):
            self.W2_cnn = weight_variable([self.filter_height_size[1], self.lstm_hidden_size*2, 1, self.filter_nums])
            self.B2_cnn = bias_variable([self.filter_nums])        
            self.conv2 = tf.nn.relu(tf.nn.conv2d(tf.expand_dims(self.lstm_final, axis=-1), self.W2_cnn, strides = [1,1,self.lstm_hidden_size*2,1], padding='SAME') + self.B2_cnn)    ##(?, 40, 1, 512)
            self.conv2_dp = tf.nn.dropout(self.conv2, self.droup_keep_prob)
            
        with tf.name_scope("cnn3"):
            self.W3_cnn = weight_variable([self.filter_height_size[2], self.lstm_hidden_size*2, 1, self.filter_nums])
            self.B3_cnn = bias_variable([self.filter_nums])
            self.conv3 = tf.nn.relu(tf.nn.conv2d(tf.expand_dims(self.lstm_final, axis=-1), self.W3_cnn, strides = [1,1,self.lstm_hidden_size*2,1], padding='SAME') + self.B3_cnn)    ##(?, 40, 1, 512)
            self.conv3_dp = tf.nn.dropout(self.conv3, self.droup_keep_prob)
            self.conv = tf.concat([self.conv1_dp,self.conv2_dp,self.conv3_dp], axis=3)
            print self.conv.shape
         
        with tf.name_scope("pooling"):
            self.pool = tf.nn.max_pool( self.conv, ksize=[1, self.n_steps, 1, 1], strides=[1, self.n_steps, 1, 1], padding='SAME')    ##(?, 100, 1, 192)
            print self.pool.shape
            self.pool = tf.reshape(self.pool, [-1, self.filter_nums*3])
            
        with tf.name_scope('softmax'):
            self.W_softm = weight_variable([self.filter_nums*3, self.num_class])
            self.B_softm = bias_variable([self.num_class])            
            self.prediction = tf.nn.softmax(tf.matmul(self.pool, self.W_softm) + self.B_softm)
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_one_hot * tf.log(self.prediction + (1e-10) ), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        
        with tf.name_scope('accuracy'):
            correct_predict = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y_one_hot, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predict,"float"),name = "accuracy")
            
    
    
    
    
    
    
import tensorflow as tf
import numpy as np
import cPickle
import random
import math
import sys
sys.path.append("..")
from lstm_two_cell_model_h import model
from lstm_two_cell_model_h import load_data
from lstm_two_cell_model_h import transfer_labels
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def one_flod(train_data, train_labels, test_data, test_labels):
    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        m = model(data, num_class, parameter_configs)
        with tf.Session(config = tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            print tf.trainable_variables ()
            data_generate_train = load_data(train_data, train_labels, batch_size)
            max_test_accuracy = 0
            test_accuracy_list = []
            for step in range(16250):
                input_x, input_y = data_generate_train.next()
                feed_dict = {
                             m.x: input_x,
                             m.x_minus: time_minus(input_x),
                             m.y: input_y,
                             m.droup_keep_prob: 0.9,
                             m.x_keep_prob:1.0,
                             m.on_train: True
                             }
                train_step,  acc, loss = sess.run([m.train_step, m.acc, m.cross_entropy], feed_dict)

                if step % 275 == 0:
                    print "train_step: %d, train_acc: %f, loss: %f"%(step,acc,loss)
                    test_accuracy = m.acc.eval(feed_dict={m.x: test_data, m.x_minus: time_minus(test_data), m.y: test_labels, m.droup_keep_prob: 1.0,m.x_keep_prob:1.0, m.on_train: False})
                    test_accuracy_list.append(test_accuracy)
                    print "test_acc:", test_accuracy,
                    print "max_test_acc:", max_test_accuracy
                    if test_accuracy > max_test_accuracy:
                        max_test_accuracy = test_accuracy
                    
    return max(test_accuracy_list[-10:])
    
def time_minus(data_matrix):
    temp_zeros = np.zeros((data_matrix.shape[0], 1, data_matrix.shape[2]))
    data_matrix_temp = np.column_stack((temp_zeros, data_matrix))
    data_matrix_temp = np.delete(data_matrix_temp, -1, axis=1)
    data_matrix_temp = data_matrix - data_matrix_temp
    data_matrix_temp = np.delete(data_matrix_temp, 0, axis=1)
    return np.column_stack((temp_zeros, data_matrix_temp))
    
if __name__ == '__main__':
    
    fold_nums = 10
    batch_size = 32
    test_batch_size = 32
    num_class = 10
    parameter_configs = {"filter_height_size":[2, 3, 5], 
                         "filter_nums":96,
                         "learning_rate":1e-3,
                         "lstm_hidden_size":128
                          }    
    print('Loading data...')	
    filepath = './dataset/ASD.p'
    data, labels, _ = cPickle.load(open(filepath, 'rb'))
    
    print('Transfering labels...')
    labels, num_classes = transfer_labels(labels)
    
    nb_samples = data.shape[0]
    print "nb_samples", nb_samples

    count = 0
    data_tmp = np.zeros(data.shape)
    labels_tmp = np.zeros(labels.shape)    
    
    li = range(nb_samples)
    random.shuffle(li)
    for i in li:
        data_tmp[count] = data[i]
        labels_tmp[count] = labels[i]
        count += 1
    data = data_tmp
    labels = labels_tmp
    
    L = [x for x in range(nb_samples)]
    lis = []
    p = 0
    for i in range(fold_nums):
        if i == fold_nums - 1:
            lis.append(L[p:])
        else:
            lis.append(L[p:p + int(nb_samples/fold_nums)])
        p = p + int(nb_samples/fold_nums)
    
    L = []
    for i in range(fold_nums):
        l = []
        l.extend(lis[i])
        for j in range(fold_nums):
            if j != i:
                l.extend(lis[j])
        L.append(l)

                
    all_fold_last = []
    for n in range(fold_nums):
        num = 0
        test_data = np.zeros((len(lis[n]), data.shape[1], data.shape[2]))
        test_labels = np.zeros(len(lis[n]))
        for m in range(len(lis[n])):
            test_data[m] = data[L[n][num]]
            test_labels[m] = labels[L[n][num]]
            num = num + 1

        train_data = np.zeros((nb_samples - len(lis[n]), data.shape[1], data.shape[2]))
        train_labels = np.zeros(nb_samples - len(lis[n]))
        train_length = np.zeros(nb_samples - len(lis[n]))
        for m in range(nb_samples - len(lis[n])):
            train_data[m] = data[L[n][num]]
            train_labels[m] = labels[L[n][num]]
            num = num + 1
            
        test_nums = test_data.shape[0]
        test_batch_nums = int(math.ceil(test_nums / (test_batch_size + 0.0)))
        
        last_accuracy = one_flod(train_data, train_labels, test_data, test_labels)
        all_fold_last.append(last_accuracy)
        
    print "------------------------------------------------"
    print "avg_last_test_acc:", sum(all_fold_last)/(len(all_fold_last)+0.0)
    
    
    
from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np
import pickle
import networkx as nx
import sys
from model import *
import time
import os
import argparse

#########################################
parser = argparse.ArgumentParser(description='cvae')

## required
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-zd', type=int, help='z dim')
parser.add_argument('-lr', type=float, help='learning rate')


args = parser.parse_args()

zd = args.zd
hd = args.hd
lr = args.lr



#########################
# parameters
#########################
N = 114
batch_size = 64

n_neg = 200

x_dim = N
z_dim = zd
c_dim = x_dim
h_encode = [hd]
h_decode = h_encode[::-1]

# train data
train_file = './data/seq_train.pickle'
valid_file = './data/seq_test.pickle'

random_seed = 123
restore_ckpt = True

n_samples = 100 # num of samples for prediction

print_iter = 20

#topk_list = [1, 3, 5, 10, 20]
topk = 1

save_path = './checkpoints/model_%s_%s_%s' % (zd, hd, lr)
restore_path ='./checkpoints/model_%s_%s_%s' % (zd, hd, lr)


###################################
# graph
########################
tf.reset_default_graph()


#negative samples
with open('../../datasets/group-recom/enron/neg_samples_test.pickle', 'rb') as file:
    neg_samples = pickle.load(file)


with open(valid_file, 'rb') as file:
    valid = pickle.load(file)



valid_obj = Input_neg(valid, neg_samples, N, n_neg, do_shuffle=False)

global_step = tf.Variable(0, name='global_step', trainable=False)
increase_global_step = tf.assign(global_step, global_step+1)

x_full = tf.placeholder(tf.float32, shape=[None, N])
x_observed = tf.placeholder(tf.float32, shape=[None, N])
x_unobserved = tf.placeholder(tf.float32, shape=[None, N])


z = tf.placeholder(tf.float32, shape=[None, z_dim])



prob_valid, logits_valid = make_decoder(z, x_observed, x_dim, c_dim, h_decode, z_dim, seed=random_seed, scope='decoder')


# init
init_op = tf.global_variables_initializer()

# saver
saver = tf.train.Saver()


start_time = time.time()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run(init_op)

    if restore_ckpt:
        saver.restore(sess, restore_path)
        print("Parameters restored from file: %s" % restore_path)


    # validation
    good_sample = {}
    for step in range(valid_obj.epoch_size):
        valid_full, valid_observed, valid_unobserved, samples = valid_obj.next_batch()
        prob_list = []
        for _ in range(n_samples):
            prob = sess.run(prob_valid, feed_dict={x_observed: valid_observed, z: np.random.normal(size=(1,z_dim))})

            prob_list.append(prob[0])

        prob_mean = np.mean(prob_list, axis=0)

        prob_ = prob_mean.copy()


        scores = []
        for i,s in enumerate(samples):
            scores.append((i, np.dot(prob_, s) / np.sum(s)))


        ss = sorted(scores, key=lambda (x,y): -y)

        ind_cand = list(zip(*ss)[0])

        good_sample[step] = prob_

    valid_obj.shuffle()

with open('./case_study/test_sample.pickle', 'wb') as file:
    pickle.dump(good_sample, file)


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


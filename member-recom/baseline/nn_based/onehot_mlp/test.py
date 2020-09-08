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
parser = argparse.ArgumentParser(description='mlp')

## required
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-lr', type=float, help='learning rate')


args = parser.parse_args()


hd = args.hd
lr = args.lr


#########################
# parameters
#########################
N = 1946
batch_size = 64
valid_batch_size = 1

x_dim = N
h_dim = [hd, N]

valid_file = './data/seq_test.pickle'

random_seed = 123
restore_ckpt = True

num_epoch = 30

print_iter = 20

#lr = 1e-3

#valid_step = 10
top = 10

restore_path ='./checkpoints/model_%s_%s' % (hd, lr)


###################################
# graph
########################
tf.reset_default_graph()

# load data
with open(valid_file, 'rb') as file:
    valid = pickle.load(file)


with tf.device('/device:GPU:0'):
    valid_obj = Input(valid, N, valid_batch_size, do_shuffle=False)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    increase_global_step = tf.assign(global_step, global_step+1)

    x_full = tf.placeholder(tf.float32, shape=[None, N])
    x_observed = tf.placeholder(tf.float32, shape=[None, N])
    x_unobserved = tf.placeholder(tf.float32, shape=[None, N])

    prob, logits = mlp(x_observed, x_dim, h_dim, seed=random_seed, scope='mlp')

    #loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x_unobserved), 1))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=x_unobserved))


    optimize = tf.train.AdamOptimizer(lr).minimize(loss)

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
    hit_rate = []
    for step in range(valid_obj.epoch_size):
        _, valid_observed, valid_unobserved = valid_obj.next_batch()
        pred = sess.run(prob, feed_dict={x_observed: valid_observed})
        for i in range(valid_batch_size):
            hit_rate.append(hit_at(pred[i], valid_unobserved[i], top))

    with open('./results/hit_at%s_h%s_lr%s_test.txt' % (top, hd, lr), 'w') as file:
        file.write('%f\n' % (np.mean(hit_rate)))

    valid_obj.shuffle()



end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


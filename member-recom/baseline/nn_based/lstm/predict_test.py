from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import collections
import os
import datetime as dt
import pickle
from model import *
import time
import argparse
#########################################
parser = argparse.ArgumentParser(description='lstm')

## required
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-lr', type=float, help='learning rate')
parser.add_argument('-ts', type=int, help='time step')


args = parser.parse_args()

hd = args.hd
lr = args.lr
ts = args.ts

top = [5,10,20,50,100]
hit_rate = {k: [] for k in top}

#burn_out = 10 # after this time_step the results are used to estimate hit rate
burn_out = ts-1

MAX_NUM_AUTH = 10
N = 1946
time_step = ts
batch_size = 1
num_epochs = 10

num_layers = 2
hidden_size = hd
embedd_dim = hidden_size

do_shuffle_authors = False


event_path = './data/seq_ts%s_test.pickle' % (ts)
author_path = './data/seq_email_ts%s_test.pickle' % (ts)
seqlen_path = './data/seq_len_ts%s_test.pickle' % (ts)
model_path = 'checkpoint/seq_ts%s_hd%s_lr%s' % (ts,hd,lr)
embedding_path = 'checkpoint/embedding_ts%s_hd%s_lr%s.pickle' % (ts,hd,lr)



learning_rate = lr
max_lr_epoch = 100
lr_decay = 0.9


dropout=0.5
init_scale=0.05

is_training = False
print_iter = 10


#################
# initialization
#################
tf.reset_default_graph()

with open(embedding_path, 'r') as file:
    embedd = pickle.load(file)

with open(event_path, 'rb') as file:
    event = pickle.load(file)

with open(author_path) as file:
    authors = pickle.load(file)

with open(seqlen_path) as file:
    seqlen = pickle.load(file)


input_obj = Input(event, authors, seqlen, batch_size, do_shuffle_authors)

lower_triangular_ones = tf.constant(np.tril(np.ones([MAX_NUM_AUTH,MAX_NUM_AUTH])),dtype=tf.float32)

m = Model(embedd, lower_triangular_ones, is_training, batch_size, time_step, N,
            hidden_size, num_layers,
            dropout=dropout, init_scale=init_scale)


init_op = tf.global_variables_initializer()

saver = tf.train.Saver()


##############
# testing
###################
with tf.Session() as sess:
    # restore the trained model
    saver.restore(sess, model_path)

    for step in range(input_obj.epoch_size):
        current_state = np.zeros((num_layers, 2, batch_size, hidden_size))

        next_example, next_label, next_len = input_obj.next_batch()

        dist = sess.run(m.softmax_out, feed_dict={m.x:next_example, m.y:next_label, m.z:next_len, m.init_state: current_state})

        pred_dist = dist[burn_out:]
        target = next_label[-1,burn_out:]
        l = next_len[-1,burn_out:]

	for i in range(len(pred_dist)):
            t = target[i]
            known = next_example[-1,burn_out:,:l[i]-1][i]
            pred_dist[i][known] = 0.0
            for top_k in top:
	        p = pred_dist[i].argsort()[::-1][:top_k]
                if t in p:
                    hit_rate[top_k].append(1.0)
                else:
                    hit_rate[top_k].append(0.0)

    with open('./results/hit_ts%s_hd%s_lr%s_test.txt' % (ts, hd, lr), 'w') as file:
        for top_k in top:
            file.write('hit at %s: %.3f\n' % (top_k, np.mean(hit_rate[top_k])))


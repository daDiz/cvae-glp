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


MAX_NUM_AUTH = 10
N = 1946
time_step = ts
batch_size = 32
num_epochs = 20

num_layers = 2
hidden_size = hd
embedd_dim = hidden_size

do_shuffle_authors = True

event_path = './data/seq_ts%s_train.pickle' % (ts)
author_path = './data/seq_email_ts%s_train.pickle' % (ts)
seqlen_path = './data/seq_len_ts%s_train.pickle' % (ts)
model_path = 'checkpoint/seq_ts%s_hd%s_lr%s' % (ts,hd,lr)
embedding_path = 'checkpoint/embedding_ts%s_hd%s_lr%s.pickle' % (ts,hd,lr)


learning_rate = lr
max_lr_epoch = 90
lr_decay = 0.9

dropout=0.5
init_scale=0.05

is_training = True
print_iter = 10

#####################
# create folders
#####################
if not os.path.exists('checkpoint/'):
    os.makedirs('checkpoint/')

if not os.path.exists('cst/'):
    os.makedirs('cst/')

if not os.path.exists('results/'):
    os.makedirs('results/')


######################
# initialization
#####################
tf.reset_default_graph()

with open(event_path, 'rb') as file:
    event = pickle.load(file)

with open(author_path) as file:
    authors = pickle.load(file)

with open(seqlen_path) as file:
    seqlen = pickle.load(file)


with tf.device('/device:GPU:0'):
    input_obj = Input(event, authors, seqlen, batch_size, do_shuffle_authors)

    lower_triangular_ones = tf.constant(np.tril(np.ones([MAX_NUM_AUTH,MAX_NUM_AUTH])),dtype=tf.float32)

    embedd = tf.random_uniform([N, embedd_dim], -init_scale, init_scale)

    m = Model(embedd, lower_triangular_ones, is_training, batch_size, time_step, N,
            hidden_size, num_layers,
            dropout=dropout, init_scale=init_scale)

    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay

    saver = tf.train.Saver()


#######################
# training
#########################
start = time.time()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run([init_op])
    cst = []
    for epoch in range(num_epochs):
        new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
        m.assign_lr(sess, learning_rate * new_lr_decay)

        for step in range(input_obj.epoch_size):
            current_state = np.zeros((num_layers, 2, batch_size, hidden_size))

            next_example, next_label, next_len = input_obj.next_batch() # [batch_size, time_step, MAX_NUM_AUTH]

            if step % print_iter != 0:
                cost, _ = sess.run([m.cost, m.train_op],
                feed_dict={m.x:next_example, m.y:next_label, m.z: next_len, m.init_state: current_state})
            else:
                cost, _, acc = sess.run([m.cost, m.train_op, m.accuracy],
                feed_dict={m.x:next_example, m.y:next_label, m.z: next_len, m.init_state: current_state})

                print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}".format(epoch, step, cost, acc))
                cst.append(cost)

        input_obj.shuffle()

    # do a final save
    saver.save(sess, model_path)

    # write embedding
    with open(embedding_path, 'wb') as file:
        pickle.dump(sess.run(m.embedding), file)

    with open('./cst/cst_ts%s_hd%s_lr%s.pickle' % (ts,hd,lr), 'wb') as file:
        pickle.dump(np.array(cst), file)

end = time.time()
print('total elapsed time: %s sec' % (end - start))

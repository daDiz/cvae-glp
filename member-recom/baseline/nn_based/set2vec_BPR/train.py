from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np
import pickle
import networkx as nx
import sys
from model import *
import time
import os
from collections import defaultdict
import argparse

#########################################
parser = argparse.ArgumentParser(description='set2vec_bpr')

## required
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-ts', type=int, help='time step')
parser.add_argument('-lr', type=float, help='learning rate')


args = parser.parse_args()

ts = args.ts
hd = args.hd
lr = args.lr


#########################
# parameters
#########################
N = 1946

batch_size = 64

valid_all = True

batch_size_valid = 1

k_embed = hd # embedding dimension
max_len = 10
timesteps = ts
n_neg = 20

top = 10
top_list = [5,10,20,50,100]


# train data
data_file = './data/seq_train.pickle'
length_file = './data/seq_length_train.pickle'
freq_file = './data/freq.pickle'

# valid data
data_file_valid = './data/seq_valid.pickle'
length_file_valid = './data/seq_length_valid.pickle'

# embedding
embed_file = './data/embed_ts%s_hd%s_lr%s.pickle' % (ts, hd, lr)

# cost
cst_file = './cst/cst_ts%s_hd%s_lr%s.pickle' % (ts, hd, lr)

# hit
hit_file = './results/hit_rate_ts%s_hd%s_lr%s.txt' % (ts, hd, lr)


restore_ckpt = False
trainable = True

num_epoch = 30

print_iter = 10


init_scale = 0.05

save_path = './checkpoints/model_ts%s_hd%s_lr%s' % (ts, hd, lr)
restore_path ='./checkpoints/model_ts%s_hd%s_lr%s' % (ts, hd, lr)

#####################
# create folders
#####################
if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

if not os.path.exists('cst/'):
    os.makedirs('cst/')

if not os.path.exists('results/'):
    os.makedirs('results/')


###################################
# graph
########################
tf.reset_default_graph()

# train data
with open(data_file, 'rb') as file:
    data = pickle.load(file)

with open(length_file, 'rb') as file:
    length = pickle.load(file)

with open(freq_file, 'rb') as file:
    freq = pickle.load(file)

# validation data
with open(data_file_valid, 'rb') as file:
    data_valid = pickle.load(file)

with open(length_file_valid, 'rb') as file:
    length_valid = pickle.load(file)

# embedding
embed = tf.random_uniform([N, k_embed], -init_scale, init_scale)


with tf.device('/device:GPU:0'):
    input_obj = Input(data, length, freq, batch_size, max_len,
                    N, n_neg,
                    shuffle_row=True, shuffle_col=True)

    valid_obj = Input_valid(data_valid, length_valid, batch_size_valid, max_len, shuffle_row=True, shuffle_col=True)


    embedding = tf.get_variable('embedding', shape=[N, k_embed], initializer=tf.random_uniform_initializer(), dtype=tf.float32, trainable=True)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    increase_global_step = tf.assign(global_step, global_step+1)

    ind = tf.placeholder(tf.int32, shape=[None, max_len])
    mask = tf.placeholder(tf.float32, shape=[None, max_len])

    ind_pos = tf.placeholder(tf.int32, shape=[None, 1]) # postive index
    ind_neg = tf.placeholder(tf.int32, shape=[None, n_neg]) # negative indices

    # [batch_size, max_len, k_embed]
    input_set = tf.nn.embedding_lookup(embedding, ind)

    # [batch_size, max_len, 1, k_embed]
    input_set_reshape = tf.reshape(input_set, [-1, max_len, 1, k_embed])

    # code_set: [batch_size, k_embed]
    _, _, _, code_set = set2vec(input_set_reshape, timesteps,
                    mprev=None, cprev=None,
                    mask=mask, inner_prod='default',
                    trainable=trainable,
                    name='lstm/')

    code_set_reshape = tf.reshape(code_set, [-1, k_embed, 1])

    # [batch_size, 1, k_embed]
    pos_set = tf.nn.embedding_lookup(embedding, ind_pos)
    # [batch_size, n_neg, k_embed]
    neg_set = tf.nn.embedding_lookup(embedding, ind_neg)

    # [-1, 1, k_embed] * [-1, k_embed, 1]
    # [-1, 1]
    pos_energies = tf.reshape(tf.matmul(pos_set, code_set_reshape),
                            [-1, 1])

    # [-1, n_neg, k_embed] * [-1, k_embed, 1]
    # [-1, n_neg]
    neg_energies = tf.reshape(tf.matmul(neg_set, code_set_reshape),
                            [-1, n_neg])

    #loss = -tf.reduce_sum(tf.reduce_mean(tf.log_sigmoid(pos_energies - neg_energies),axis=1))
    loss = -tf.reduce_sum(tf.log_sigmoid(pos_energies - neg_energies))

    solver = tf.train.AdamOptimizer(lr).minimize(loss)

    # init
    init_op = tf.global_variables_initializer()

    # saver
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='lstm')+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embedding'))


start_time = time.time()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run(init_op)

    if restore_ckpt:
        saver.restore(sess, restore_path)
        print("Parameters restored from file: %s" % restore_path)


    cst_list = []
    hit_history = []
    for epoch in range(num_epoch):
        for step in range(input_obj.epoch_size):
            indices, _, m, indices_pos, indices_neg = input_obj.next_batch()
            _, cst = sess.run([solver, loss],
            feed_dict={ind: indices, mask: m, ind_pos: indices_pos, ind_neg: indices_neg})

            if np.isnan(cst):
                raise Exception('NaN encounter ts%s hd%s lr%s' % (ts,hd,lr))

            sess.run(increase_global_step)

            cst_list.append(cst)

            if step % print_iter == 0:
                print("Epoch %s Step %s cost %f" % (epoch, step, cst))
        #valid_obj.shuffle()
        #input_obj.shuffle()

    # estimate hit rate using the entire validation set
    hit_dict = defaultdict(list)
    for step in range(valid_obj.epoch_size):
        indices_valid, _, m_valid, indices_pos_valid = valid_obj.next_batch()
        emb, query = sess.run([embedding, code_set], feed_dict={ind:indices_valid, mask:m_valid})

        for k in top_list:
            cand = hit_at(emb, query[0], k)
            if indices_pos_valid[0,0] in cand:
                hit_dict[k].append(1.0)
            else:
                hit_dict[k].append(0.0)

    print("Train completes")
    saver.save(sess, save_path)


    with open(hit_file, 'w') as file:
        for k in top_list:
            file.write('hit at %d: %f\n' % (k, np.mean(hit_dict[k]))) 

    with open(cst_file, 'wb') as file:
        pickle.dump(cst_list, file)

    with open(embed_file, 'wb') as file:
        pickle.dump(sess.run(embedding), file)



end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


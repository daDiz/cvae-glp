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


# test data
data_file_valid = './data/seq_test.pickle'
length_file_valid = './data/seq_length_test.pickle'

# embedding
embed_file = './data/embed_ts%s_hd%s_lr%s.pickle' % (ts, hd, lr)

# cost
cst_file = './cst/cst_ts%s_hd%s_lr%s.pickle' % (ts, hd, lr)

# hit
hit_file = './results/hit_rate_ts%s_hd%s_lr%s_test.txt' % (ts, hd, lr)


restore_ckpt = True
trainable = False

num_epoch = 30

print_iter = 20


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


###################################
# graph
########################
tf.reset_default_graph()

# validation data
with open(data_file_valid, 'rb') as file:
    data_valid = pickle.load(file)

with open(length_file_valid, 'rb') as file:
    length_valid = pickle.load(file)


with tf.device('/device:GPU:0'):
    valid_obj = Input_valid(data_valid, length_valid, batch_size_valid, max_len, shuffle_row=True, shuffle_col=True)


    embedding = tf.get_variable('embedding', shape=[N, k_embed], initializer=tf.random_uniform_initializer(), dtype=tf.float32, trainable=trainable)



    ind = tf.placeholder(tf.int32, shape=[None, max_len])
    mask = tf.placeholder(tf.float32, shape=[None, max_len])


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


    with open(hit_file, 'w') as file:
        for k in top_list:
            file.write('hit at %d: %f\n' % (k, np.mean(hit_dict[k]))) 


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


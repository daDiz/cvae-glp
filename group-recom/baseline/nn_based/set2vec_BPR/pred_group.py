from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np
import pickle
import networkx as nx
import sys
from model import *
from pred_group_util import *
import time
from collections import defaultdict
import argparse

#########################################
parser = argparse.ArgumentParser(description='attBPR')

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
N = 114

batch_size = 64

valid_all = True

batch_size_valid = 1

k_embed = hd # embedding dimension
max_len = 36
timesteps = ts
#n_neg = 200


#top = 10
top_list = [1,3,5,10,20]
#top_list = [1]


# test data
data_file_valid = './data/seq_valid.pickle'
length_file_valid = './data/seq_length_valid.pickle'

# hit
hit_file = './results/hit_rate_ts%s_hd%s_lr%s_group_valid.txt' % (ts, hd, lr)


restore_ckpt = True
trainable = False


print_iter = 10


init_scale = 0.05

save_path = './checkpoints/model_ts%s_hd%s_lr%s' % (ts, hd, lr)
restore_path ='./checkpoints/model_ts%s_hd%s_lr%s' % (ts, hd, lr)


###################################
# graph
########################
tf.reset_default_graph()

#negative samples
with open('../../../../datasets/group-recom/enron/neg_samples_valid.pickle', 'rb') as file:
    neg_samples = pickle.load(file)


# validation data
with open(data_file_valid, 'rb') as file:
    data_valid = pickle.load(file)

with open(length_file_valid, 'rb') as file:
    length_valid = pickle.load(file)


valid_obj = Input_group(neg_samples, data_valid, length_valid, max_len, shuffle_row=False, shuffle_col=False)


embedding = tf.get_variable('embedding', shape=[N, k_embed], initializer=tf.random_uniform_initializer(), dtype=tf.float32, trainable=trainable)



ind = tf.placeholder(tf.int32, shape=[None, max_len])
mask = tf.placeholder(tf.float32, shape=[None, max_len])


# [batch_size, max_len, k_embed]
input_set = tf.nn.embedding_lookup(embedding, ind)

# [batch_size, max_len, 1, k_embed]
input_set_reshape = tf.reshape(input_set, [-1, max_len, 1, k_embed])

# code_set: [batch_size, k_embed]
att, _, _, code_set = set2vec(input_set_reshape, timesteps,
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
        indices_valid, length_valid, m_valid = valid_obj.next_batch()
        n_neg = len(indices_valid)-1
        emb, attribute, query = sess.run([embedding, att, code_set], feed_dict={ind:indices_valid, mask:m_valid})

        ind_pos = indices_valid[-1][length_valid[-1]-1]
        for k in top_list:
            cand = hit_at_group(query, emb[ind_pos], k)
            if n_neg in cand:
                hit_dict[k].append(1.0)
            else:
                hit_dict[k].append(0.0)


    with open(hit_file, 'w') as file:
        for k in top_list:
            file.write('hit at %d: %f\n' % (k, np.mean(hit_dict[k])))


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


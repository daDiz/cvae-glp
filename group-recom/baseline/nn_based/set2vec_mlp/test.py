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
parser = argparse.ArgumentParser(description='set2vec-mlp')

## required
parser.add_argument('-ed', type=int, help='embedding dim')
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-lr', type=float, help='learning rate')
parser.add_argument('-ts', type=int, help='time step')

args = parser.parse_args()

ed = args.ed
hd = args.hd
lr = args.lr
ts = args.ts

#########################
# parameters
#########################
N = 114
batch_size = 64
batch_size_valid = 1
k_embed = ed # embedding dimension
max_len = 36
timesteps = ts

topk_list = [1,3,5,10,20]

n_neg = 200

x_dim = k_embed

h_mlp = [hd, N]

embed_file = './data/embed_ed%s_hd%s_lr%s_ts%s.pickle' % (ed, hd, lr, ts)


# valid data
data_valid_file = './data/seq_test.pickle'
length_valid_file = './data/seq_length_test.pickle'


#negative samples
with open('../../../../datasets/group-recom/enron/neg_samples_test.pickle', 'rb') as file:
    neg_samples = pickle.load(file)


restore_ckpt = True

num_epoch = 30

print_iter = 20

random_seed = 123

init_scale = 0.05

save_path = './checkpoints/model_ed%s_hd%s_lr%s_ts%s' % (ed, hd, lr, ts)
restore_path ='./checkpoints/model_ed%s_hd%s_lr%s_ts%s'% (ed, hd, lr, ts)


###################################
# graph
########################
tf.reset_default_graph()

# load data
with open(data_valid_file, 'rb') as file:
    data_valid = pickle.load(file)

with open(length_valid_file, 'rb') as file:
    length_valid = pickle.load(file)


with open(embed_file, 'rb') as file:
    embed = pickle.load(file)

#embed = tf.random_uniform([N, k_embed], -init_scale, init_scale)


valid_obj = Input_neg(data_valid, length_valid, neg_samples, N, max_len, n_neg, shuffle_row=False, shuffle_col=True)

embedding = tf.Variable(initial_value=embed, dtype=tf.float32, trainable=True, name='embedding')

global_step = tf.Variable(0, name='global_step', trainable=False)
increase_global_step = tf.assign(global_step, global_step+1)

ind_valid = tf.placeholder(tf.int32, shape=[None, max_len])

y = tf.placeholder(tf.float32, shape=[None, N])
y_valid = tf.placeholder(tf.float32, shape=[None, N])


mask_valid = tf.placeholder(tf.float32, shape=[None, max_len])

# [batch_size, max_len, k_embed]
valid_set = tf.nn.embedding_lookup(embedding, ind_valid)

# [batch_size, max_len, 1, k_embed]
valid_set_reshape = tf.reshape(valid_set, [-1, max_len, 1, k_embed])

# code_set: [batch_size, k_embed]
_, _, _, code_set_valid = set2vec(valid_set_reshape, timesteps,
                mprev=None, cprev=None,
                mask=mask_valid, inner_prod='default',
                name='lstm')

prob, _ = mlp(code_set_valid, k_embed, h_mlp, seed=random_seed, scope='mlp')



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
    hit_rate = {k: [] for k in topk_list}
    emb = sess.run(embedding)
    for step in range(valid_obj.epoch_size):
        indices_valid, m_valid, samples = valid_obj.next_batch()
        pred = sess.run(prob,
                        feed_dict={ind_valid: indices_valid, mask_valid: m_valid})

        prob_ = pred[0]

        scores = []
        for i, s in enumerate(samples):
            scores.append((i, np.dot(prob_, s)/np.sum(s)))

        ss = sorted(scores, key=lambda (x,y): -y)
        ind_cand = list(zip(*ss)[0])

        for topk in topk_list:
            if n_neg in ind_cand[:topk]:
                hit_rate[topk].append(1.)
            else:
                hit_rate[topk].append(0.)

    valid_obj.shuffle()

    with open('./results/hit_test_ed%s_hd%s_lr%s_ts%s.txt' % (ed, hd, lr, ts), 'w') as file:
        for topk in topk_list:
            file.write('hit at %d: %f\n' % (topk, np.mean(hit_rate[topk])))


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


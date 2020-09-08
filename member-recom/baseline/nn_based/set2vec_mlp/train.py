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
N = 1946
batch_size = 64
batch_size_valid = 1
k_embed = ed # embedding dimension
max_len = 10
timesteps = ts

top = 10

x_dim = k_embed

h_mlp = [hd, N]

# train data
data_file = './data/seq_train.pickle'
length_file = './data/seq_length_train.pickle'
embed_file = './data/embed_ed%s_hd%s_lr%s_ts%s.pickle' % (ed, hd, lr, ts)
cst_file = './cst/cst_ed%s_hd%s_lr%s_ts%s.pickle' % (ed, hd, lr, ts)

# valid data
data_valid_file = './data/seq_valid.pickle'
length_valid_file = './data/seq_length_valid.pickle'


restore_ckpt = False

num_epoch = 30

print_iter = 20


random_seed = 123

init_scale = 0.05

save_path = './checkpoints/model_ed%s_hd%s_lr%s_ts%s' % (ed, hd, lr, ts)
restore_path ='./checkpoints/model_ed%s_hd%s_lr%s_ts%s'% (ed, hd, lr, ts)

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

# load data
with open(data_file, 'rb') as file:
    data = pickle.load(file)

with open(length_file, 'rb') as file:
    length = pickle.load(file)


with open(data_valid_file, 'rb') as file:
    data_valid = pickle.load(file)

with open(length_valid_file, 'rb') as file:
    length_valid = pickle.load(file)


#with open(embed_file, 'rb') as file:
#    embed = pickle.load(file)

embed = tf.random_uniform([N, k_embed], -init_scale, init_scale)


with tf.device('/device:GPU:0'):
    input_obj = Input(data, length, batch_size, max_len,
                    shuffle_row=False, shuffle_col=True)
    valid_obj = Input(data_valid, length_valid, batch_size_valid, max_len,
                    shuffle_row=False, shuffle_col=True)

    embedding = tf.Variable(initial_value=embed, dtype=tf.float32, trainable=True, name='embedding')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    increase_global_step = tf.assign(global_step, global_step+1)

    ind = tf.placeholder(tf.int32, shape=[None, max_len]) # positive indices
    ind_valid = tf.placeholder(tf.int32, shape=[None, max_len])

    y = tf.placeholder(tf.int32, shape=[None])
    y_valid = tf.placeholder(tf.int32, shape=[None])

    y_onehot = tf.one_hot(y, N)

    mask = tf.placeholder(tf.float32, shape=[None, max_len])
    mask_valid = tf.placeholder(tf.float32, shape=[None, max_len])

    # [batch_size, max_len, k_embed]
    input_set = tf.nn.embedding_lookup(embedding, ind)
    valid_set = tf.nn.embedding_lookup(embedding, ind_valid)

    # [batch_size, max_len, 1, k_embed]
    input_set_reshape = tf.reshape(input_set, [-1, max_len, 1, k_embed])
    valid_set_reshape = tf.reshape(valid_set, [-1, max_len, 1, k_embed])

    # code_set: [batch_size, k_embed]
    _, _, _, code_set = set2vec(input_set_reshape, timesteps,
                    mprev=None, cprev=None,
                    mask=mask, inner_prod='default',
                    name='lstm')

    _, _, _, code_set_valid = set2vec(valid_set_reshape, timesteps,
                    mprev=None, cprev=None,
                    mask=mask_valid, inner_prod='default',
                    name='lstm')

    _, logits = mlp(code_set, k_embed, h_mlp, seed=random_seed, scope='mlp')

    prob, _ = mlp(code_set_valid, k_embed, h_mlp, seed=random_seed, scope='mlp')


    #loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
    #labels=y_onehot), 1))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_onehot))


    solver = tf.train.AdamOptimizer(lr).minimize(loss)

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


    cst_list = []
    for epoch in range(num_epoch):
        for step in range(input_obj.epoch_size):
            indices, m, target = input_obj.next_batch()
            _, cst = sess.run([solver, loss], feed_dict={ind: indices, mask: m, y: target})


            sess.run(increase_global_step)

            cst_list.append(cst)

            if step % print_iter == 0:
                print("Epoch %s Step %s cost %f" % (epoch, step, cst))

        input_obj.shuffle()

    # validation
    hit_rate = []
    emb = sess.run(embedding)
    for step in range(valid_obj.epoch_size):
        indices_valid, m_valid, target_valid = valid_obj.next_batch()
        pred = sess.run(prob,
                        feed_dict={ind_valid: indices_valid, mask_valid: m_valid})
        for i in range(batch_size_valid):
            hit_rate.append(hit_at(pred[i], target_valid[i], top))

    with open('./results/hit_at%s_ed%s_hd%s_lr%s_ts%s.txt' % (top, ed, hd, lr, ts), 'w') as file:
        file.write('%f\n' % (np.mean(hit_rate)))

    valid_obj.shuffle()

    print("Train completes")
    saver.save(sess, save_path)


    with open(cst_file, 'wb') as file:
        pickle.dump(cst_list, file)

    with open(embed_file, 'wb') as file:
        pickle.dump(sess.run(embedding), file)


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


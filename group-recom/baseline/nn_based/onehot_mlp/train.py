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
N = 114
batch_size = 64
valid_batch_size = 1

n_neg = 200

x_dim = N
h_dim = [hd, N]

# train data
train_file = './data/seq_train.pickle'
valid_file = './data/seq_valid.pickle'

#negative samples
with open('../../../../datasets/group-recom/enron/neg_samples_valid.pickle', 'rb') as file:
    neg_samples = pickle.load(file)

cst_file = './cst/cst_hd%s_lr%s.pickle' % (hd, lr)

random_seed = 123
restore_ckpt = False

num_epoch = 30

print_iter = 20

#lr = 1e-3

#valid_step = 10
#top = 10

topk_list = [1,3,5,10,20]

save_path = './checkpoints/model_%s_%s' % (hd, lr)
restore_path ='./checkpoints/model_%s_%s' % (hd, lr)

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
with open(train_file, 'rb') as file:
    data = pickle.load(file)

with open(valid_file, 'rb') as file:
    valid = pickle.load(file)


with tf.device('/device:GPU:0'):
    input_obj = Input(data, N, batch_size, do_shuffle=False)
    valid_obj = Input_neg(valid, neg_samples, N, n_neg, do_shuffle=False)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    increase_global_step = tf.assign(global_step, global_step+1)

    x_full = tf.placeholder(tf.float32, shape=[None, N])
    x_observed = tf.placeholder(tf.float32, shape=[None, N])
    x_unobserved = tf.placeholder(tf.float32, shape=[None, N])

    prob, logits = mlp(x_observed, x_dim, h_dim, seed=random_seed, scope='mlp')

    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x_unobserved), 1))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=x_unobserved))


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


    cst_list = []
    for epoch in range(num_epoch):
        for step in range(input_obj.epoch_size):
            data_full, data_observed, data_unobserved = input_obj.next_batch()
            _, cst = sess.run([optimize, loss], feed_dict={x_observed:data_observed, x_unobserved:data_unobserved})

            sess.run(increase_global_step)

            cst_list.append(cst)

            if step % print_iter == 0:
                print("Epoch %s Step %s cost %f" % (epoch, step, cst))

        input_obj.shuffle()

    # validation
    hit_rate = {k: [] for k in topk_list}
    for step in range(valid_obj.epoch_size):
        valid_full, valid_observed, valid_unobserved, samples = valid_obj.next_batch()
        prob_valid = sess.run(prob, feed_dict={x_observed:valid_observed})

        prob_ = prob_valid[0]

        scores = []
        for i,s in enumerate(samples):
            scores.append((i, np.dot(prob_, s) / np.sum(s)))

        ss = sorted(scores, key=lambda (x,y): -y)

        ind_cand = list(zip(*ss)[0])

        for topk in topk_list:
            if n_neg in ind_cand[:topk]:
                hit_rate[topk].append(1.)
            else:
                hit_rate[topk].append(0.)

    valid_obj.shuffle()


    with open('./results/hit_h%s_lr%s_valid.txt' % (hd, lr), 'w') as file:
        for topk in topk_list:
            file.write('hit at %d: %f\n' % (topk, np.mean(hit_rate[topk])))

    print("Train completes")
    saver.save(sess, save_path)

    with open(cst_file, 'wb') as file:
        pickle.dump(cst_list, file)


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


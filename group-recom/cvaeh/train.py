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
import math

#########################################
parser = argparse.ArgumentParser(description='cvae-ts')

## required
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-zd', type=int, help='z dim')
parser.add_argument('-lr', type=float, help='learning rate')
parser.add_argument('-ts', type=int, help='history window size, not including the current step')

args = parser.parse_args()

zd = args.zd
hd = args.hd
lr = args.lr
ts = args.ts


#########################
# parameters
#########################
N = 114
batch_size = 64

x_dim = N
z_dim = zd
c1_dim = x_dim
c2_dim = x_dim

h_encode = [hd]
h_decode = h_encode[::-1]

# train data
train_file = './data/seq_train.pickle'


# history
hist_train_file = './data/hist_train_%s.pickle' % (ts)


cst_file = './cst/cst_%s_%s_%s_%s.pickle' % (zd, hd, ts, lr)

random_seed = 123
restore_ckpt = False

#num_epoch = 200
num_epoch = 20

n_samples = 100 # num of samples for prediction

print_iter = 20

save_path = './checkpoints/model_%s_%s_%s_%s' % (zd, hd, ts, lr)
restore_path ='./checkpoints/model_%s_%s_%s_%s' % (zd, hd, ts, lr)

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


# load history
with open(hist_train_file, 'rb') as file:
    hist_train = pickle.load(file)


with tf.device('/device:GPU:0'):
    input_obj = Input(data, hist_train, N, batch_size, do_shuffle=False)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    increase_global_step = tf.assign(global_step, global_step+1)

    x_full = tf.placeholder(tf.float32, shape=[None, N])
    x_observed = tf.placeholder(tf.float32, shape=[None, N])
    x_unobserved = tf.placeholder(tf.float32, shape=[None, N])

    hs = tf.placeholder(tf.float32, shape=[None, N])

    posterior_mu, posterior_logvar = make_encoder(x_full, x_observed, hs, x_dim, c1_dim, c2_dim, h_encode, z_dim, seed=random_seed, scope='posterior')

    z_sample_posterior = sample_z(posterior_mu, posterior_logvar)

    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    _, logits = make_decoder(z_sample_posterior, x_observed, hs, x_dim, c1_dim, c2_dim, h_decode, z_dim, seed=random_seed, scope='decoder')

    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x_full), 1)

    kl_loss = 0.5 * tf.reduce_sum(tf.exp(posterior_logvar) + posterior_mu**2 - 1. - posterior_logvar, 1)

    vae_loss = tf.reduce_mean(recon_loss +  kl_loss)

    optimize = tf.train.AdamOptimizer(lr).minimize(vae_loss)

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
            data_full, data_observed, data_unobserved, data_hs = input_obj.next_batch()

            _, cst = sess.run([optimize, vae_loss], feed_dict={x_full:data_full,
            x_observed:data_observed, x_unobserved:data_unobserved, hs: data_hs})

            sess.run(increase_global_step)

            cst_list.append(cst)

            if step % print_iter == 0:
                print("Epoch %s Step %s cost %f" % (epoch, step, cst))

        input_obj.shuffle()

    print("Train completes")
    saver.save(sess, save_path)

    with open(cst_file, 'wb') as file:
        pickle.dump(cst_list, file)


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


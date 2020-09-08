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
parser = argparse.ArgumentParser(description='cvae')

## required
parser.add_argument('-hd', type=int, help='hidden dim')
parser.add_argument('-zd', type=int, help='z dim')
parser.add_argument('-lr', type=float, help='learning rate')


args = parser.parse_args()

zd = args.zd
hd = args.hd
lr = args.lr



#########################
# parameters
#########################
N = 1946
batch_size = 64
valid_batch_size = 1

x_dim = N
z_dim = zd
c_dim = x_dim
h_encode = [hd]
h_decode = h_encode[::-1]

valid_file = './data/seq_test.pickle'

random_seed = 123
restore_ckpt = True

num_epoch = 100

n_samples = 100 # num of samples for prediction

print_iter = 20

top = 10

save_path = './checkpoints/model_%s_%s_%s' % (zd, hd, lr)
restore_path ='./checkpoints/model_%s_%s_%s' % (zd, hd, lr)


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

    posterior_mu, posterior_logvar = make_encoder(x_full, x_observed, x_dim, c_dim, h_encode, z_dim, seed=random_seed, scope='posterior')

    z_sample_posterior = sample_z(posterior_mu, posterior_logvar)

    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    _, logits = make_decoder(z_sample_posterior, x_observed, x_dim, c_dim, h_decode, z_dim, seed=random_seed, scope='decoder')

    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x_full), 1)

    kl_loss = 0.5 * tf.reduce_sum(tf.exp(posterior_logvar) + posterior_mu**2 - 1. - posterior_logvar, 1)

    vae_loss = tf.reduce_mean(recon_loss +  kl_loss)

    samples, _ = make_decoder(z, x_observed, x_dim, c_dim, h_decode, z_dim, seed=random_seed, scope='decoder')

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


    # validation
    hit_rate = []
    for step in range(valid_obj.epoch_size):
        _, valid_observed, valid_unobserved = valid_obj.next_batch()
        samples_test = []
        for k in range(n_samples):
            samples_test.append(sess.run(samples, feed_dict={z: np.random.normal(size=(valid_batch_size, z_dim)), x_observed: valid_observed}))
        samples_test = np.mean(samples_test, axis=0)
        for i in range(valid_batch_size):
            hit_rate.append(hit_at(valid_observed[i], valid_unobserved[i], samples_test[i], top))

    with open('./results/hit_at%s_z%s_h%s_lr%s_test.txt' % (top, zd, hd, lr), 'w') as file:
        file.write('%f\n' % (np.mean(hit_rate)))

    valid_obj.shuffle()


end_time = time.time()
print('total elapsed time: %s sec' % (end_time - start_time))


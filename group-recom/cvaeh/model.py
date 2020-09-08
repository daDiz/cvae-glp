from __future__ import print_function, absolute_import, division
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import networkx as nx
import pickle
import random
from random import shuffle

tfd = tfp.distributions

class Input(object):
    def __init__(self, data, history, N, batch_size, do_shuffle=False):
        self.data = data
        self.history = history
        self.N = N
        self.batch_size = batch_size

        self.num_epochs = 0

        self.n_samples = len(self.data)
        self.epoch_size = self.n_samples // self.batch_size

        self.do_shuffle = do_shuffle

        self.cur = 0

    def shuffle(self):
        self.cur = 0
        self.num_epochs += 1

        if self.do_shuffle:
            shuffle(self.data)

    def next_batch(self):
        if self.cur + self.batch_size > self.n_samples:
            raise Exception('epoch exhausts')

        x = self.data[self.cur:self.cur+self.batch_size]
        x_full = np.zeros([self.batch_size, self.N], dtype=float)
        x_observed = np.zeros([self.batch_size, self.N], dtype=float)
        x_unobserved = np.zeros([self.batch_size, self.N], dtype=float)

        hs = self.history[self.cur:self.cur+self.batch_size]

        for i in range(self.batch_size):
            shuffle(x[i]) # shuffle indices
            x_full[i, x[i]] = 1.0
            x_observed[i, x[i][:-1]] = 1.0
            x_unobserved[i, x[i][-1]] = 1.0

        self.cur += self.batch_size


        return x_full, x_observed, x_unobserved, hs


class Input_neg(object):
    def __init__(self, data, history, neg_samples, N, n_neg, do_shuffle=False):
        self.data = data
        self.history = history
        self.neg_samples = neg_samples

        self.N = N
        self.batch_size = 1

        self.n_neg = n_neg

        self.num_epochs = 0

        self.n_samples = len(self.data)
        self.epoch_size = self.n_samples // self.batch_size

        self.do_shuffle = do_shuffle

        self.cur = 0

    def shuffle(self):
        self.cur = 0
        self.num_epochs += 1

        if self.do_shuffle:
            shuffle(self.data)

    # 0 ~ n_neg are negative samples
    # n_neg is the positive sample
    def next_batch(self):
        if self.cur + self.batch_size > self.n_samples:
            raise Exception('epoch exhausts')

        x = self.data[self.cur:self.cur+self.batch_size]
        x_full = np.zeros([self.batch_size, self.N], dtype=float)
        x_observed = np.zeros([self.batch_size, self.N], dtype=float)
        x_unobserved = np.zeros([self.batch_size, self.N], dtype=float)

        hs = np.reshape(self.history[self.cur:self.cur+self.batch_size], (1,-1))

        pos_ind = x[0][-1]
        neg_ind = self.neg_samples[pos_ind][0]
        neg_len = self.neg_samples[pos_ind][1]

        samples = np.zeros([self.batch_size+self.n_neg, self.N], dtype=float)

        for i in range(self.n_neg):
            samples[i, neg_ind[i][:neg_len[i]-1]] = 1.0


        samples[self.n_neg, x[0][:-1]] = 1.0

        x_full[0, x[0]] = 1.0
        x_observed[0, pos_ind] = 1.0
        x_unobserved[0, x[0][:-1]] = 1.0

        self.cur += self.batch_size


        return x_full, x_observed, x_unobserved, hs, samples


class Input_neg_old(object):
    def __init__(self, data, history, neg_samples, N, n_neg, do_shuffle=False):
        self.data = data
        self.history = history
        self.neg_samples = neg_samples

        self.N = N
        self.batch_size = 1

        self.n_neg = n_neg

        self.num_epochs = 0

        self.n_samples = len(self.data)
        self.epoch_size = self.n_samples // self.batch_size

        self.do_shuffle = do_shuffle

        self.cur = 0

    def shuffle(self):
        self.cur = 0
        self.num_epochs += 1

        if self.do_shuffle:
            shuffle(self.data)

    # 0 ~ n_neg are negative samples
    # n_neg is the positive sample
    def next_batch(self):
        if self.cur + self.batch_size > self.n_samples:
            raise Exception('epoch exhausts')

        x = self.data[self.cur:self.cur+self.batch_size]
        x_full = np.zeros([self.batch_size+self.n_neg, self.N], dtype=float)
        x_observed = np.zeros([self.batch_size+self.n_neg, self.N], dtype=float)
        x_unobserved = np.zeros([self.batch_size+self.n_neg, self.N], dtype=float)

        hs = self.history[self.cur:self.cur+self.batch_size]
        hs_all = np.repeat(hs,self.batch_size+self.n_neg,axis=0)

        pos_ind = x[0][-1]
        neg_ind = self.neg_samples[pos_ind][0]
        neg_len = self.neg_samples[pos_ind][1]

        for i in range(self.n_neg):
            x_full[i, neg_ind[i][:neg_len[i]-1]] = 1.0
            x_full[i, pos_ind] = 1.0
            x_unobserved[i, neg_ind[i][:neg_len[i]-1]] = 1.0
            x_observed[i, pos_ind] = 1.0

        x_full[self.n_neg, x[0]] = 1.0
        x_observed[self.n_neg, pos_ind] = 1.0
        x_unobserved[self.n_neg, x[0][:-1]] = 1.0

        self.cur += self.batch_size

        return x_full, x_observed, x_unobserved, hs_all


# x : [-1, n_input]
def dense(x, n_input, n_output, seed, scope='dense'):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		w1 = tf.get_variable(name='w1', shape=[n_input, n_output],
							initializer=tf.glorot_uniform_initializer(seed=seed, dtype=tf.float32))

		b1 = tf.get_variable(name='b1', shape=[n_output],
							initializer=tf.zeros_initializer(dtype=tf.float32))

	x1 = tf.matmul(x, w1) + b1

	return x1


def make_prior(z_dim):
    loc = tf.zeros(z_dim)
    scale = tf.ones(z_dim)
    return tfd.MultivariateNormalDiag(loc, scale)


# x : [-1, k_embed]
# c : [-1, k_embed]
# gcn_dim : [k-1, n1, n2, n3]
# h_dim: [n_hidden1, n_hidden2, ...]
def make_encoder(x, c1, c2, x_dim, c1_dim, c2_dim, h_dim, z_dim, seed=123, scope='encoder'):
    x1 = tf.concat([x,c1,c2], axis=1)
    x1 = dense(x1,
            x_dim+c1_dim+c2_dim,
            h_dim[0],
            seed=seed,
            scope=scope+'/dense0')

    x1 = tf.nn.relu(x1)

    for i in range(len(h_dim)-1):
        x1 = dense(x1,
                h_dim[i],
                h_dim[i+1],
                seed=seed,
                scope=scope+'/dense%d'%(i+1))

        x1 = tf.nn.relu(x1)

    gaussian_params = dense(x1,
                            h_dim[-1],
                            z_dim*2,
                            seed=seed,
                            scope=scope+'/dense%d'%(len(h_dim)))

    mu = tf.slice(gaussian_params,[0,0],[-1,z_dim]) # [-1,latent]
    log_var = tf.slice(gaussian_params, [0,z_dim],[-1,-1])
    return mu, log_var


# z : [-1, latent]
# h_dim: [n_hidden_1, n_hidden_2, ...]
def make_decoder(z, c1, c2, x_dim, c1_dim, c2_dim, h_dim, z_dim, seed=123, scope='decoder'):

    x1 = tf.concat([z, c1, c2], axis=1)
    x1 = dense(x1,
            z_dim+c1_dim+c2_dim,
            h_dim[0],
            seed=seed,
            scope=scope+'/dense0')

    x1 = tf.nn.relu(x1)

    for i in range(len(h_dim)-1):
        x1 = dense(x1,
                h_dim[i],
                h_dim[i+1],
                seed=seed,
                scope=scope+'/dense%d'%(i+1))

        x1 = tf.nn.relu(x1)

    logits = dense(x1,
                h_dim[-1],
                x_dim,
                seed=seed,
                scope=scope+'/dense%d'%(len(h_dim)))

    prob = tf.nn.sigmoid(logits)

    return prob, logits

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# return top results
def hit_at(x_ob, x_unob, x_pred, top):
    x_pred[x_ob == 1.] = 0.
    x_ind = np.argsort(x_pred)[::-1]
    top_x = x_ind[:top]
    target = np.argmax(x_unob)
    if target in top_x:
        return 1.0
    else:
        return 0.0


if __name__=='__main__':
    N = 113
    batch_size = 3

    data_file = './data/seq_train.pickle'
    hs_file = './data/hist_train_8.pickle'

    # load data
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    with open(hs_file, 'rb') as file:
        hist = pickle.load(file)

    input_obj = Input(data, hist, N, batch_size, do_shuffle=False)

    x_full, x_ob, x_unob, hs = input_obj.next_batch()
    print(x_full)
    print(x_ob)
    print(x_unob)
    print(hs)

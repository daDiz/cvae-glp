from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import os
import networkx as nx
import pickle
import random
from random import shuffle


class Input(object):
    def __init__(self, data, N, batch_size, do_shuffle=False):
        self.data = data
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

        for i in range(self.batch_size):
            shuffle(x[i]) # shuffle indices
            x_full[i, x[i]] = 1.0
            x_unobserved[i, x[i][:-1]] = 1.0
            x_observed[i, x[i][-1]] = 1.0

        self.cur += self.batch_size


        return x_full, x_observed, x_unobserved

class Input_neg(object):
    def __init__(self, data, neg_samples, N, n_neg, do_shuffle=False):
        self.data = data
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


        return x_full, x_observed, x_unobserved, samples


# x : [-1, n_input]
def dense(x, n_input, n_output, seed, scope='dense'):
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		w1 = tf.get_variable(name='w1', shape=[n_input, n_output],
							initializer=tf.glorot_uniform_initializer(seed=seed, dtype=tf.float32))

		b1 = tf.get_variable(name='b1', shape=[n_output],
							initializer=tf.zeros_initializer(dtype=tf.float32))

	x1 = tf.matmul(x, w1) + b1

	return x1


def mlp(x, x_dim, h_dim, seed=123, scope='mlp'):
    x1 = dense(x,
            x_dim,
            h_dim[0],
            seed=seed,
            scope=scope+'/dense0')

    x1 = tf.nn.relu(x1)

    for i in range(len(h_dim)-2):
        x1 = dense(x1,
                h_dim[i],
                h_dim[i+1],
                seed=seed,
                scope=scope+'/dense%d'%(i+1))

        x1 = tf.nn.relu(x1)

    logits = dense(x1,
                h_dim[-2],
                h_dim[-1],
                seed=seed,
                scope=scope+'/dense%d'%(len(h_dim)))

    prob = tf.nn.sigmoid(logits)
    #prob = tf.nn.softmax(logits)

    return prob, logits

# return top results
def hit_at(x_pred, x_unob,top):
    x_ind = np.argsort(x_pred)[::-1]
    top_x = x_ind[:top]
    target = np.argmax(x_unob)
    if target in top_x:
        return 1.0
    else:
        return 0.0


# return top results
def hit_at_sigmoid(x_ob, x_unob, x_pred, top):
    x_pred[x_ob == 1.] = 0.
    x_ind = np.argsort(x_pred)[::-1]
    top_x = x_ind[:top]
    target = np.argmax(x_unob)
    if target in top_x:
        return 1.0
    else:
        return 0.0


if __name__=='__main__':
    N = 3153
    batch_size = 6

    data_file = './data/seq_seen_2000_2001_train.pickle'

    # load data
    with open(data_file, 'rb') as file:
        data = pickle.load(file)


    input_obj = Input(data, N, batch_size, do_shuffle=False)

    example = input_obj.next_batch()
    print(example[:,:10])

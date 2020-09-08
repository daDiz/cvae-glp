from __future__ import print_function, absolute_import, division
import tensorflow as tf
import numpy as np
import networkx as nx
import pickle
import random
from random import shuffle
import math

class Input_group(object):
    def __init__(self, neg_samples, data, length, max_len, shuffle_row=False, shuffle_col=False):

        self.neg_samples = neg_samples
        self.data = data
        self.length = length

        self.max_len = max_len # max length of a sample

        self.num_epochs = 0

        self.n_samples = len(self.data)
        self.epoch_size = self.n_samples

        self.shuffle_row = shuffle_row
        self.shuffle_col = shuffle_col

        self.cur = 0

        self.lower_ones = np.tril(np.ones((max_len, max_len)))
        self.diag_ones = np.eye(self.max_len, dtype=int)

    def shuffle(self):
        self.cur = 0
        self.num_epochs += 1

        if self.shuffle_row:
            idx = np.arange(self.n_samples, dtype=int)
            random.shuffle(idx)
            self.data = self.data[idx]
            self.length = self.length[idx]

        if self.shuffle_col:
            for i in range(self.n_samples):
                l = self.length[i]
                idx = np.arange(l, dtype=int)
                random.shuffle(idx)
                self.data[i][:l] = self.data[i][idx]

    def next_batch(self):
        if self.cur >= self.n_samples:
            self.shuffle()

        x = np.reshape(self.data[self.cur],(1,-1))
        x_len = self.length[self.cur]
        mask = np.take(self.lower_ones, [x_len-2], axis=0)

        x_pos = self.data[self.cur][x_len-1]
        #mask_pos = np.take(self.diag_ones, [x_len-1], axis=0)
        #x_pos = np.sum(x * mask_pos)

        x_len = np.reshape(x_len, [-1,1])

        # select some random negative samples
        #ind_neg = np.random.choice(self.N, size=self.n_neg, replace=False)

        x_neg = self.neg_samples[x_pos][0]
        x_len_neg = self.neg_samples[x_pos][1]

        #x_neg = np.take(self.data_pool, ind_neg, axis=0)
        #x_len_neg = self.length_pool[ind_neg]
        mask_neg = np.take(self.lower_ones, x_len_neg-2, axis=0)

        x_len_neg = np.reshape(x_len_neg, [-1,1])

        #sample_ind = np.array(list(ind_neg) + [self.cur])
        sample = np.concatenate((x_neg, x),axis=0)
        sample_len = np.reshape(np.concatenate((x_len_neg, x_len),axis=0), [-1])
        sample_mask = np.concatenate((mask_neg, mask),axis=0)

        #idx = list(range(self.n_neg+1))
        
        #random.shuffle(idx)

        #sample_ind = sample_ind[idx] 
        #sample = sample[idx]
        #sample_len = sample_len[idx]
        #sample_mask = sample_mask[idx]
    
        self.cur += 1

        return sample, sample_len, sample_mask #, x_pos, self.cur-1


# return top results
def hit_at_group(groups, x, top):
    d = [(i, groups[i]) for i in range(len(groups))]
    d_sort = sorted(d, key=lambda e : -np.dot(e[1],x))
 
    return [d_sort[i][0] for i in range(top)]



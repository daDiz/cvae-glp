from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import math
import os
import argparse
#########################################
parser = argparse.ArgumentParser(description='lstm')

## required
parser.add_argument('-ts', type=int, help='time step')


args = parser.parse_args()

ts = args.ts

if not os.path.exists('data/'):
    os.makedirs('data/')


def get_seq(n, time_step):
    seq = []
    n_seq = n-time_step+1
    for i in range(n_seq):
        win = []
        for j in range(time_step):
            win.append(i+j)

        seq.append(win)

    return seq


# generate a sequence of authors
# each window contains time_step of samples
# each sample contains multiple authors (at least two)
# authors are represented using indices
def build_seq(in_name, out_seq, out_author, out_len, max_len, time_step, sep=','):
    ll = []
    auth = []
    with open(in_name, 'r') as in_file:
        for line in in_file:
            elems = line.strip('\n').split(sep)
            authors = list(map(int, elems[1:]))
            l = len(authors)
            ll.append(l)
            if l < max_len:
                auth.append(authors + [0] * (max_len - l))
            else:
                auth.append(authors)




    m = len(auth)
    seq = get_seq(m, time_step)

    with open(out_seq, 'wb') as file:
        pickle.dump(np.array(seq), file)

    # auth - [n_samples, max_len]
    with open(out_author, 'wb') as file:
        pickle.dump(np.array(auth), file)

    # auth - [n_samples]
    with open(out_len, 'wb') as file:
        pickle.dump(np.array(ll), file)




time_step = ts
max_len = 10
for x in ['train', 'valid', 'test']:
    build_seq('../../../../datasets/member-recom/enron/seq_%s.txt' % (x),
        'data/seq_ts%s_%s.pickle' % (ts,x),
        'data/seq_email_ts%s_%s.pickle' % (ts,x),
        'data/seq_len_ts%s_%s.pickle' % (ts,x),
        time_step=time_step, max_len=max_len, sep=',')


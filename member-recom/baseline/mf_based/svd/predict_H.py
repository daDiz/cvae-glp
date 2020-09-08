import pickle
import numpy as np
from scipy import spatial
import argparse
from collections import defaultdict

#########################################
parser = argparse.ArgumentParser(description='svd')

## required
parser.add_argument('-s', type=str, help='valid or test')
parser.add_argument('-m', type=str, help='distance, cos or euc')


args = parser.parse_args()


flag = args.s
method = args.m
top = [5,10,20,50,100]


def pred_H(W, H, x, ind, start, method='cos'):
    h = H[start+ind]
    x_ = x[ind]
    dist = []
    for j in range(len(W)):
        if j not in x_:
            if method == 'cos':
                d = spatial.distance.cosine(W[j], h)
            elif method == 'euc':
                d = spatial.distance.euclidean(W[j], h)
            else:
                raise Exception('unknown method, cos or euc')
        else:
            d = np.float('inf')

        dist.append(d)

    r = np.argsort(dist)
    return r


if flag == 'valid':
    start = 24957
    end = start + 3112
elif flag == 'test':
    start = 24957 + 3112
    end = 24957 + 3112 + 3076
else:
    raise Exception('uknown data set. valid or test')


with open('./data/W_train.pickle', 'rb') as file:
    W = pickle.load(file)

with open('./data/H_train.pickle', 'rb') as file:
    H = pickle.load(file).T

with open('./data/x_%s.pickle' % (flag), 'rb') as file:
    x = pickle.load(file)

with open('./data/y_%s.pickle' % (flag), 'rb') as file:
    y = pickle.load(file)


hit = defaultdict(list)

for i, target in enumerate(y):
    if start + i >= end:
        raise Exception('i should be in [start, end]')

    p = pred_H(W, H, x, i, start, method)

    for t in top:
        if target in p[:t]:
            hit[t].append(1.)
        else:
            hit[t].append(0.)

for t in top:
    print('hit at %d : %f' % (t, np.mean(hit[t])))


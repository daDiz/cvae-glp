import pickle
import numpy as np
from scipy import spatial
import argparse
from collections import defaultdict
import random
import os

if not os.path.exists('results/'):
    os.makedirs('results/')


#########################
def compare_with_ties(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return random.randint(0, 1) * 2 - 1


#########################################
parser = argparse.ArgumentParser(description='svd')

## required
parser.add_argument('-s', type=str, help='valid or test')
#parser.add_argument('-m', type=str, help='distance, cos or euc')
parser.add_argument('-ncp', type=int, help='num components')


args = parser.parse_args()

n_components = args.ncp



flag = args.s
#method = args.m
method = 'cos'
if flag == 'valid':
    start = 12640
    end = start + 1580
elif flag == 'test':
    start = 12640 + 1580
    end = 12640 + 1580 + 1581
else:
    raise Exception('uknown data set. valid or test')


#top = 10
top_list = [5,10,20]
n_neg = 200
aggs = ['sum', 'mean', 'max', 'min']


def pred_H(W, H, x, pos_idx, neg_idx, method='cos'):
    w = W[x]

    dist = []

    n_neg = len(neg_idx)

    if n_neg != 200:
        raise Exception('n_neg != 200')

    for i, j in enumerate(neg_idx):
        if method == 'cos':
            #d = spatial.distance.cosine(H[j], w)
            d = np.dot(H[j],w)
        elif method == 'euc':
            d = spatial.distance.euclidean(H[j], w)
        else:
            raise Exception('unknown method, cos or euc')

        dist.append((i,d))

    if method == 'cos':
        #d = spatial.distance.cosine(H[pos_idx], w)
        d = np.dot(H[pos_idx],w)
    elif method == 'euc':
        d = spatial.distance.cosine(H[pos_idx], w)
    else:
        raise Exception('unknown method')

    dist.append((n_neg, d))

    return dist



with open('./data/W.pickle', 'rb') as file:
    W = pickle.load(file)

with open('./data/H.pickle', 'rb') as file:
    H = pickle.load(file).T


with open('./data/x_%s.pickle' % (flag), 'rb') as file:
    xv = pickle.load(file)


with open('../../../../datasets/group-recom/enron/neg_idx_%s.pickle' % (flag), 'rb') as file:
    neg_idx = pickle.load(file)



hit_valid = {k: [] for k in top_list}
for i, x in enumerate(xv):
    nx = neg_idx[x]

    px = i + start

    scores = pred_H(W, H, x, px, nx, method='cos')

    ss = sorted(scores, key=lambda (x,y): -y, cmp=compare_with_ties)


    ind_cand = list(zip(*ss)[0])

    for top in top_list:
        if n_neg in ind_cand[:top]:
            hit_valid[top].append(1.)
        else:
            hit_valid[top].append(0.)



with open('./results/hit_n%s_%s.txt' % (n_components, flag), 'w') as file:
    for top in top_list:
        file.write('hit at %d: %s\n' % (top, np.mean(hit_valid[top])))



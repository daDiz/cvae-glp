##############################################################
# sum pair-wise common neighbor for group common neighbor
###############################################################
from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import networkx as nx
import sys
import argparse
from collections import defaultdict
import time
from scipy.spatial import distance
import os

if not os.path.exists('results/'):
    os.makedirs('results/')


def load_data(in_name, delimiter=','):
    data = []
    with open(in_name, 'r') as file:
        for line in file:
            elems = line.strip('\n').split(delimiter)[1:]
            data.append(map(int, elems))

    return data

def load_emb(in_name):
    data = {}
    with open(in_name, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                elems = line.strip('\n').split(' ')
                n, m = int(elems[0]), int(elems[1])
            else:
                elems = line.strip('\n').split(' ')
                k = int(elems[0])
                data[k] = map(float, elems[1:])

    return data, n, m

#########################################
parser = argparse.ArgumentParser(description='node2vec')

parser.add_argument('-f', type=str, help='valid or test')
parser.add_argument('-m', type=str, help='sum, max or min')
parser.add_argument('-d', type=str, help='dimension')
parser.add_argument('-p', type=str, help='p')
parser.add_argument('-q', type=str, help='q')


args = parser.parse_args()
flag = args.f
method = args.m
d = args.d
p = args.p
q = args.q

emb_name = './emb/enron/enron_%s_%s_%s.emb' % (d, p, q)
data_name = '../../../../datasets/member-recom/enron/seq_%s.txt' % (flag)


data = load_data(data_name, delimiter=',')

emb, n_node, n_edge = load_emb(emb_name)


topk_list = [5,10,20]

hr = {k: [] for k in topk_list}


score_dict = {} # score dictionary
cn_dict = {} # common neighbor dictionary
sec_dict = {} # second degree neighbor dictionary

start_time = time.time()

for step, x in enumerate(data):
    group = x[:-1]
    label = x[-1]

    kg = tuple(group)
    if kg in score_dict:
        score = score_dict[kg]
    else:
        score = np.zeros(n_node)

        for a in group:
            for b in range(n_node):
                if b not in group:
                    k = (a, b) if a <= b else (b, a)
                    if k in cn_dict:
                        cn = cn_dict[k]
                    else:
                        cn = distance.cosine(emb[a],emb[b]) 
                        cn_dict[k] = cn

                    if method == 'max':
                        score[b] = max(cn, score[b])
                    elif method == 'sum':
                        score[b] += cn
                    elif method == 'min':
                        if score[b] > 0:
                            score[b] = min(cn, score[b])
                        else:
                            score[b] = cn
                    else:
                        raise Exception('uknown method')

        score = np.argsort(score)

        score_dict[kg] = score

    for topk in topk_list:
        if label in score[:topk]:
            hr[topk].append(1.)
        else:
            hr[topk].append(0.)

    if step > 0 and step % 10 == 0:
        cur_time = time.time()
        print('Step %s Time %s sec\n' % (step, cur_time - start_time))
        sys.stdout.flush()

end_time = time.time()

file = open('./results/hit_rate_%s_%s_d%s_p%s_q%s.txt' % (flag, method, d, p, q), 'w')

for topk in topk_list:
    file.write('hit at %d: %f\n' % (topk, np.mean(hr[topk])))

file.close()

print('complete in %s sec' % (end_time - start_time))


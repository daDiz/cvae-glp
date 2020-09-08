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
import random
import os

if not os.path.exists('results/'):
    os.makedirs('results/')


def load_graph(in_name):
    g = nx.read_edgelist(in_name, delimiter=' ', nodetype=int)

    return g

def load_data(in_name, delimiter=','):
    data = []
    with open(in_name, 'r') as file:
        for line in file:
            elems = line.strip('\n').split(delimiter)[1:]
            data.append(map(int, elems))

    return data

# process negative groups (remove the last member in the group)
def gen_neg_groups(data, data_len):
    samples = []
    for i in range(len(data)):
        samples.append(list(data[i][:data_len[i]-1]))

    return samples

def compare_with_ties(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return random.randint(0, 1) * 2 - 1
   
#########################################
parser = argparse.ArgumentParser(description='common neighbor')

parser.add_argument('-f', type=str, help='valid or test')
parser.add_argument('-m', type=str, help='mean, sum, max or min')



args = parser.parse_args()
flag = args.f 
method = args.m

 
graph_name = '../../../../datasets/group-recom/enron/enron_edgelist.txt'
data_name = '../../../../datasets/group-recom/enron/seq_%s.txt' % (flag)


# negative samples
with open('../../../../datasets/group-recom/enron/neg_samples_%s.pickle' % (flag), 'rb') as file:
    neg_samples = pickle.load(file)


data = load_data(data_name, delimiter=',') 



g = load_graph(graph_name)


topk_list = [1,3,5,10,20]

hr = {k: [] for k in topk_list}

n = len(g.nodes)

cn_dict = {} # common neighbor dictionary
#sec_dict = {} # second degree neighbor dictionary

start_time = time.time()

for step, x in enumerate(data):
    group = x[:-1]
    b = x[-1]

    ng = neg_samples[b][0]
    ng_len = neg_samples[b][1]

    samples = gen_neg_groups(ng, ng_len)

    n_neg = len(samples)

    samples.append(group)


    score = [0] * (n_neg + 1)
    for ind, y in enumerate(samples):
        for a in y:
            k = (a, b) if a <= b else (b, a)
            if k in cn_dict:
                cn = cn_dict[k]
            else:
                cn = len(list(nx.common_neighbors(g,a,b)))
                cn_dict[k] = cn

            if method == 'max':
                score[ind] = max(cn, score[ind])
            elif method == 'sum':
                score[ind] += cn
            elif method == 'mean':
                score[ind] += cn * 1. / len(y)
            elif method == 'min':
                if score[ind] > 0:
                    score[ind] = min(cn, score[ind])
                else:
                    score[ind] = cn
            else:
                raise Exception('uknown method')

    score_ = zip(range(n_neg+1), score)
    ss = sorted(score_, key=lambda (x,y): -y, cmp=compare_with_ties)
    ind_cand = list(zip(*ss)[0])

    for topk in topk_list:
        if n_neg in ind_cand[:topk]:
            hr[topk].append(1.)
        else:
            hr[topk].append(0.)
    
    if step > 0 and step % 10 == 0:
        cur_time = time.time()
        print('Step %s Time %s sec\n' % (step, cur_time - start_time)) 
        sys.stdout.flush()
 
end_time = time.time()

file = open('./results/hit_group_%s_%s.txt' % (flag, method), 'w')

for topk in topk_list:
    file.write('hit at %d: %f\n' % (topk, np.mean(hr[topk])))

file.close()

print('complete in %s sec' % (end_time - start_time))


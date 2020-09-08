from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import networkx as nx
import sys
import argparse
import time
from collections import defaultdict
import random
import os

if not os.path.exists('results/'):
    os.makedirs('results/')


class Katz():
    def __init__(self, A, beta):
        self.A = A # adj
        self.beta = beta # damping coeff
        self.score = np.zeros(A.shape) # katz beta score

    # calc katz_beta within
    # max path length lm (included)
    def fit(self, lm):
        An = np.eye(len(self.A))
        for l in range(1,lm+1):
            An = np.dot(self.A, An)
            self.score += self.beta**l*An
    
    # calc katz_beta using (I - beta*A)^-1 - I 
    def score_inf(self):
        return np.linalg.inv(np.eye(len(self.A))-self.beta*self.A)-np.eye(len(self.A))  

    # given a group [xi, ...] and y
    # return the sum of score[xi, y]
    def group_score(self, group, ind, method='sum'):
        score = 0.
        for x in group:
            if method == 'sum':
                score += self.score[x, ind]
            elif method == 'max':
                score = max(score, self.score[x, ind])
            elif method == 'min':
                score = min(score, self.score[x, ind])
            elif method == 'mean':
                score += self.score[x, ind] * 1. / len(group)
            else:
                raise Exception('unknown method')
        return score

    # return the top k candidates to join the group
    def group_topk(self, group, topk=5, method='sum'):
        if method == 'sum':
            score = np.sum(np.take(self.score, group, axis=0), axis=0)
        elif method == 'max':
            score = np.max(np.take(self.score, group, axis=0), axis=0)
        elif method == 'min':
            score = np.min(np.take(self.score, group, axis=0), axis=0)
        else:
            raise Exception('unknown method')

        score[group] = -np.inf
        return np.argsort(score)[::-1][:topk]
        


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

def calc_hit_rate(data, katz_obj, topk=5, m='sum'):
    hit = []
    for x in data:
        label = x[-1]
        group = x[:-1]
        
        cand = katz_obj.group_topk(group, topk, method=m)

        if label in cand:
            hit.append(1.0)
        else:
            hit.append(0.0)


    return np.mean(hit)

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
parser = argparse.ArgumentParser(description='katz')

parser.add_argument('-f', type=str, help='valid or test')
parser.add_argument('-l', type=int, help='max l')
parser.add_argument('-b', type=float, help='beta')
parser.add_argument('-m', type=str, help='method: sum, max, min')

args = parser.parse_args()
flag = args.f 
beta = args.b
lm = args.l
m = args.m

graph_name = '../../../../datasets/group-recom/enron/enron_edgelist.txt'
data_name = '../../../../datasets/group-recom/enron/seq_%s.txt' % (flag)


# negative samples
with open('../../../../datasets/group-recom/enron/neg_samples_%s.pickle' % (flag), 'rb') as file:
    neg_samples = pickle.load(file)


data = load_data(data_name, delimiter=',') 


g = load_graph(graph_name)

A = nx.adjacency_matrix(g).todense()
katz = Katz(A, beta)
katz.fit(lm)

topk_list = [1,3,5,10,20]

n_neg = 200

hr = {k: [] for k in topk_list}

cn_dict = {} # common neighbor dictionary

start_time = time.time()

for step, x, in enumerate(data):
    group = x[:-1]
    b = x[-1]

    ng = neg_samples[b][0]
    ng_len = neg_samples[b][1]

    samples = gen_neg_groups(ng, ng_len)

    samples.append(group)

    score = [0] * (n_neg + 1)
    for ind, y in enumerate(samples):
        score[ind] = katz.group_score(y, b, method=m)

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

file = open('./results/hit_group_%s_%s_%s_%s.txt' % (flag, beta, lm, m), 'w')

for topk in topk_list:
    file.write('hit at %d: %f\n' % (topk, np.mean(hr[topk])))

file.close()

print('complete in %s sec' % (end_time - start_time))



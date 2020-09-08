##############################################################
# sum pair-wise adar for group adar
###############################################################
from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import networkx as nx
import sys
import argparse
from collections import defaultdict
import time
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

# g -- graph
# x -- src node
# k -- degree of neighbors
def k_degree_neighbors(g, x, k):
    nn = [x]
    res = []
    for i in range(k):
        tmp = []
        for a in nn:
            tmp += g.neighbors(a)
        
        nn = tmp
        res += tmp
    res = list(set(res))

    return res
    
#########################################
parser = argparse.ArgumentParser(description='adar')

parser.add_argument('-f', type=str, help='valid or test')
parser.add_argument('-m', type=str, help='sum, max or min')



args = parser.parse_args()
flag = args.f 
method = args.m


graph_name = '../../../../datasets/member-recom/enron/enron_edgelist.txt'
data_name = '../../../../datasets/member-recom/enron/seq_%s.txt' % (flag)


data = load_data(data_name, delimiter=',') 

g = load_graph(graph_name)


topk_list = [5,10,20,50,100]

hr = {k: [] for k in topk_list}

n = len(g.nodes)

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
        score = np.zeros(n)

        for a in group:
            #if a in sec_dict:
            #    nn = sec_dict[a]
            #else:
            #    nn = k_degree_neighbors(g, a, k=2)
            #    sec_dict[a] = nn

            for b in range(n):
                if b not in group:
                    k = (a, b) if a <= b else (b, a)
                    if k in cn_dict:
                        cn = cn_dict[k]
                    else:
                        ad = [x for x in nx.adamic_adar_index(g,[(a,b)])]
                        cn = ad[0][2]
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
 
        score = np.argsort(score)[::-1]
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

file = open('./results/hit_faster_%s_%s.txt' % (flag, method), 'w')

for topk in topk_list:
    file.write('hit at %d: %f\n' % (topk, np.mean(hr[topk])))

file.close()

print('complete in %s sec' % (end_time - start_time))


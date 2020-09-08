from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import networkx as nx
import sys
import argparse
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
    def group_score(self, group, ind):
        score = 0.
        for x in group:
            score += self.score[x, ind]

        return score

    # return the top k candidates to join the group
    def group_topk(self, group, topk=5):
        score = np.sum(np.take(self.score, group, axis=0), axis=0)
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

def calc_hit_rate(data, katz_obj, topk=5):
    hit = []
    for x in data:
        label = x[-1]
        group = x[:-1]
        
        cand = katz_obj.group_topk(group, topk)

        if label in cand:
            hit.append(1.0)
        else:
            hit.append(0.0)


    return np.mean(hit)


#########################################
parser = argparse.ArgumentParser(description='katz')

parser.add_argument('-f', type=str, help='valid or test')
parser.add_argument('-l', type=int, help='max l')
parser.add_argument('-b', type=float, help='beta')


args = parser.parse_args()
flag = args.f 
beta = args.b
lm = args.l


graph_name = '../../../../datasets/member-recom/enron/enron_edgelist.txt'
data_name = '../../../../datasets/member-recom/enron/seq_%s.txt' % (flag)


data = load_data(data_name, delimiter=',') 

g = load_graph(graph_name)

A = nx.adjacency_matrix(g).todense()
katz = Katz(A, beta)
katz.fit(lm)

file = open('./results/hit_rate_%s_%s_%s.txt' % (flag, beta, lm), 'w')

for topk in [5,10,20,50,100]:
    hr = calc_hit_rate(data, katz, topk)
    file.write('hit at %d: %f\n' % (topk, hr))

file.close()


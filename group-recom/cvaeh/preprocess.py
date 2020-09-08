from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import math
import sys
import os


def load_data(in_name,sep=' '):
    data = []
    with open(in_name, 'r') as in_file:
        for line in in_file:
            elems = line.strip('\n').split(sep)
            row = list(map(int, elems[1:]))

            data.append(row)

    return data

def gen_history(data, ws, N):
    """
    ws -- window size
    N -- num of nodes
    """
    history = []
    for i in range(len(data)):
        window = [0.] * N
        start = max(0, i-ws)
        end = i
        for j in range(start, end):
            for y in data[j]:
                window[y] += 1.

        history.append(window)

    return history


if __name__ == '__main__':
    if not os.path.exists('data/'):
        os.makedirs('data/')

    ws = int(sys.argv[1])
    N = 114
    for s in ['train', 'valid', 'test']:
        data = load_data('../../datasets/group-recom/enron/seq_%s.txt' % (s), sep=',')
        hist = gen_history(data, ws, N)

        with open('./data/seq_%s.pickle' % (s), 'wb') as file:
            pickle.dump(data, file)

        with open('./data/hist_%s_%d.pickle' % (s, ws), 'wb') as file:
            pickle.dump(hist, file)




from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import math
import sys
import os


# generate a list of list of indices
def prepare_data(in_name, out_data, sep=','):
    data = []
    with open(in_name, 'r') as in_file:
        for line in in_file:
            elems = line.strip('\n').split(sep)
            row = list(map(int, elems[1:]))

            data.append(row)

    with open(out_data, 'wb') as file:
        pickle.dump(data, file)



if __name__ == '__main__':
    if not os.path.exists('data/'):
        os.makedirs('data/')

    for s in ['train', 'valid', 'test']:
        prepare_data('../../datasets/group-recom/enron/seq_%s.txt' % (s),
        './data/seq_%s.pickle' % (s), sep=',')


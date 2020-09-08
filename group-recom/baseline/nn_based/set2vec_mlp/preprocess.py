from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import math
import sys
import os

# generate a list of list of indices
def prepare_data(in_name, out_data, out_length, max_len, sep=','):
    data = []
    length = []

    with open(in_name, 'r') as in_file:
        for line in in_file:
            elems = line.strip('\n').split(sep)
            row = list(map(int, elems[1:]))

            l = len(row)
            if l <= max_len:
                row += [0] * (max_len - l)
            else:
                raise Exception('row length %s > max length %s' % (l, max_len))

            data.append(row)
            length.append(l)


    with open(out_data, 'wb') as file:
        pickle.dump(np.array(data), file)

    with open(out_length, 'wb') as file:
        pickle.dump(np.array(length), file)


if __name__ == '__main__':
    if not os.path.exists('data/'):
        os.makedirs('data/')


    max_len = 36
    for s in ['train', 'valid', 'test']:
        prepare_data('../../../../datasets/group-recom/enron/seq_%s.txt' % (s),
        './data/seq_%s.pickle' % (s),
        './data/seq_length_%s.pickle' % (s),
        max_len,
        sep=',')




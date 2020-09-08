from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import math
import os

# hold out the last member from the valid set
# hold out the last member from the test set
def prepare_data(train_name, valid_name, test_name, n_word, n_doc, sep=','):
    n_row = n_word
    n_col = n_doc
    X = np.zeros((n_row, n_col))

    i = 0

    with open(train_name, 'r') as file:
        for line in file:
            row = line.strip('\n').split(sep)
            elems = list(map(int, row[1:]))
            if len(elems) != len(set(elems)):
                print(elems)
            for j in elems:
                X[j][i] += 1.

            i += 1

    label_valid = []
    x_valid = []
    with open(valid_name, 'r') as file:
        for line in file:
            row = line.strip('\n').split(sep)
            elems = list(map(int, row[1:-1]))

            idx = int(row[-1])
            x_valid.append(idx)
            label_valid.append(elems)

            for j in elems:
                X[j][i] += 1.

            i += 1

    x_test = []
    label_test = []
    with open(test_name, 'r') as file:
        for line in file:
            row = line.strip('\n').split(sep)
            elems = list(map(int, row[1:-1]))

            idx = int(row[-1])
            x_test.append(idx)
            label_test.append(elems)

            for j in elems:
                X[j][i] += 1.

            i += 1


    with open('./data/X_train.pickle', 'wb') as file:
        pickle.dump(X, file)

    with open('./data/x_valid.pickle', 'wb') as file:
        pickle.dump(x_valid, file)

    with open('./data/x_test.pickle', 'wb') as file:
        pickle.dump(x_test, file)

    with open('./data/y_valid.pickle', 'wb') as file:
        pickle.dump(label_valid, file)

    with open('./data/y_test.pickle', 'wb') as file:
        pickle.dump(label_test, file)


if __name__ == '__main__':
    n_word = 114
    n_doc = 15801

    if not os.path.exists('data/'):
        os.makedirs('data/')


    prepare_data('../../../../datasets/group-recom/enron/seq_train.txt',
    '../../../../datasets/group-recom/enron/seq_valid.txt',
    '../../../../datasets/group-recom/enron/seq_test.txt',
    n_word,
    n_doc,
    sep=',')


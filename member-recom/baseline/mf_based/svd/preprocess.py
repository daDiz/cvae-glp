from __future__ import division, print_function, absolute_import
import numpy as np
import pickle
import math
import os

# hold out a random member from the valid set
# hold out the last member from the test set
def prepare_data(in_name, n_word, n_doc, sep=','):
    n_row = n_word
    n_col = n_doc
    X = np.zeros((n_row, n_col))

    train_name = in_name + '_train.txt'
    valid_name = in_name + '_valid.txt'
    test_name = in_name + '_test.txt'

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
            elems = list(map(int, row[1:]))

            l = len(elems)
            select = np.random.randint(0,l)

            elems_ = elems[:select] + elems[select+1:]

            x_valid.append(elems_)
            label_valid.append(elems[select])

            if elems[select] in elems_:
                print((select, l, elems, elems[select], elems_))

            for j in elems_:
                X[j][i] += 1.

            i += 1

    x_test = []
    label_test = []
    with open(test_name, 'r') as file:
        for line in file:
            row = line.strip('\n').split(sep)
            elems = list(map(int, row[1:-1]))

            x_test.append(elems)
            label_test.append(int(row[-1]))

            for j in elems:
                X[j][i] += 1.

            i += 1


    with open('./data/X_holdout.pickle', 'wb') as file:
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
    n_word = 1946
    n_doc = 31145

    if not os.path.exists('data/'):
        os.makedirs('data/')


    prepare_data('../../../../datasets/member-recom/enron/seq',
    n_word,
    n_doc, sep=',')


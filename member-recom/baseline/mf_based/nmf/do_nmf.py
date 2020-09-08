import pickle
import numpy as np
from sklearn.decomposition import NMF
import argparse

#########################################
parser = argparse.ArgumentParser(description='nmf')

## required
parser.add_argument('-ncp', type=int, help='num components')
parser.add_argument('-alpha', type=float, help='alpha')

parser.add_argument('-l1', type=float, help='l1 ratio')

args = parser.parse_args()


n_components = args.ncp
alpha = args.alpha
l1_ratio = args.l1


data_file = './data/X_holdout.pickle'
W_out = './data/W_train.pickle'
H_out = './data/H_train.pickle'

init = 'nndsvda'


with open(data_file, 'rb') as file:
    data = pickle.load(file)



model = NMF(n_components=n_components,
            init=init,
            random_state=123,
            solver='mu',
            tol = 1e-4,
            max_iter = 1000,
            alpha = alpha,
            l1_ratio = l1_ratio,
            verbose=True)

W = model.fit_transform(data)
H = model.components_

print('reconstruction err %.3f' % (model.reconstruction_err_))
print('num of iter %d' % (model.n_iter_))

with open(W_out, 'wb') as file:
    pickle.dump(W, file)

with open(H_out, 'wb') as file:
    pickle.dump(H, file)



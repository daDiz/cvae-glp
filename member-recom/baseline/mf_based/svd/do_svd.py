import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
import argparse

#########################################
parser = argparse.ArgumentParser(description='svd')

## required
parser.add_argument('-ncp', type=int, help='num components')


args = parser.parse_args()


n_components = args.ncp


data_file = './data/X_holdout.pickle'
W_out = './data/W_train.pickle'
H_out = './data/H_train.pickle'

init = 'nndsvda'


with open(data_file, 'rb') as file:
    data = pickle.load(file)



model = TruncatedSVD(n_components=n_components,
            random_state=123,
            tol = 1e-4,
            n_iter = 1000)

W = model.fit_transform(data)
H = model.components_

print('explained variance ratio:')
print(model.explained_variance_ratio_)

with open(W_out, 'wb') as file:
    pickle.dump(W, file)

with open(H_out, 'wb') as file:
    pickle.dump(H, file)



import numpy as np
import pickle

thres = 0.1

with open('test_sample.pickle','rb') as file:
    sample = pickle.load(file)


with open('../../../datasets/group-recom/enron/dict_idx_role.pickle', 'rb') as file:
    idx_role = pickle.load(file)


data = []
with open('../../../datasets/group-recom/enron/seq_test.txt', 'r') as file:
    for line in file:
        elems = line.strip('\n').split(',')[1:]
        data.append(int(elems[-1]))

example = []
for ind in sample:
    src = data[ind]
    prob = sample[ind]

    key_people = np.where(prob > thres)[0]

    if len(key_people) >= 1:
        A = idx_role[src]
        B = [idx_role[k] for k in key_people if k != src]
        B_ = []
        for x in B:
            if x != 'xxx' and len(x.split('/'))>=2:
                B_.append(x)

        if A != 'xxx' and len(A.split('/'))>=2 and len(B_) > 0:
            example.append((A, B_))

with open('example_test.txt', 'w') as file:
    for x in example:
        file.write('%s,%s\n' % (x[0], ','.join(x[1])))



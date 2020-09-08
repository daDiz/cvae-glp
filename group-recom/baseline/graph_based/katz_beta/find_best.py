import numpy as np
import sys

def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(np.float(line.strip('\n').split(' ')[-1]))

    return data


top = int(sys.argv[1])
topdict = {5:0, 10:1, 20:2} 
k = topdict[top]

param = []
hit = []
for l in [1, 2, 3, 4, 5]:
    for b in [0.1,0.01,0.001,0.0001]:
        for m in ['sum', 'max', 'min', 'mean']:
            try:
                name = './results/hit_group_valid_%s_%s_%s.txt' % (b, l, m)
                r = read_file(name)[k]
                param.append('l: %s b: %s method: %s' % (l, b, m))
                hit.append(r)
            except:
                pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

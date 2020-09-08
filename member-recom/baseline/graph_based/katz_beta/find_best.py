import numpy as np
import sys

def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(np.float(line.strip('\n').split(' ')[-1]))

    return data


top = int(sys.argv[1])
topdict = {5:0, 10:1, 20:2, 50:3, 100:4} 
k = topdict[top]

param = []
hit = []
for l in [1, 2, 3, 4, 5]:
    for b in [0.1,0.01,0.001,0.0001]:
        try:
            name = './results/hit_rate_valid_%s_%s.txt' % (b, l)

            r = read_file(name)[k]
            param.append('l%s b%s' % (l, b))
            hit.append(r)
        except:
            pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

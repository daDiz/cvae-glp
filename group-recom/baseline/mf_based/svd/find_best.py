import numpy as np
import sys

def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(np.float(line.strip('\n').split(' ')[-1]))

    return data


top = int(sys.argv[1])

top_list = [5, 10, 20]

ind = top_list.index(top)

param = []
hit = []
for n in [16, 32, 64]:
    name = './results/H/hit_n%s_valid.txt' % (n)

    try:
        r = read_file(name)[ind]
        param.append('n%s' % (n))
        hit.append(r)
    except:
        pass


ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

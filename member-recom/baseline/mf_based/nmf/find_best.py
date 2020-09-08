import numpy as np
import sys

def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(np.float(line.strip('\n').split(' ')[-1]))

    return data


top = int(sys.argv[1])

top_list = [5, 10, 20, 50, 100]

ind = top_list.index(top)

param = []
hit = []
for n in [64, 128, 256, 512, 1024]:
    for a in [0.0, 0.1, 1.0, 10.0]:
        for l in [0.1, 0.5, 0.9]:
            name = './results/H/hit_n%s_a%.1f_l%.1f_valid_cos.txt' % (n, a, l)

            try:
                r = read_file(name)[ind]
                param.append('n%s a%s l%s' % (n,a,l))
                hit.append(r)
            except:
                pass


ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

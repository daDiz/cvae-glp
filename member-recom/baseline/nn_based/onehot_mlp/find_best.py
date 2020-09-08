import numpy as np
import sys

def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(np.float(line.strip('\n').split(' ')[-1]))

    return data


top = int(sys.argv[1])


param = []
hit = []
for hd in [64, 128, 256, 512, 1024]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        name = './results/hit_at%s_h%s_lr%s.txt' % (top, hd, lr)

        r = read_file(name)[0]
        param.append('hd%s lr%s' % (hd, lr))
        hit.append(r)

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

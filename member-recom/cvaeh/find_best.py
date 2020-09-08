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
for ts in [2, 4, 8, 16, 32]:
    for zd in [32, 64, 128, 256]:
        for hd in [128, 256, 512, 1024]:
            for lr in [0.00001, 0.0001, 0.001, 0.01]:
                try:
                    name = './results/hit_at%s_z%s_h%s_t%s_lr%s.txt' % (top, zd, hd, ts, lr)

                    r = read_file(name)[0]
                    param.append('zd%s hd%s ts%s lr%s' % (zd, hd, ts, lr))
                    hit.append(r)
                except:
                    pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

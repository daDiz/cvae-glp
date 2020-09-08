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
for ts in [3, 5, 10]:
    for hd in [128, 256, 512]:
        for lr in [0.0001,0.001,0.01]:
            try:
                name = './results/hit_rate_ts%s_hd%s_lr%s.txt' % (ts, hd, lr)

                r = read_file(name)[k]
                param.append('hd%s ts%s lr%s' % (hd, ts, lr))
                hit.append(r)
            except:
                pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

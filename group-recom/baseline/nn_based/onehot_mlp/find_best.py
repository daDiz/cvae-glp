import numpy as np
import sys

def read_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(np.float(line.strip('\n').split(' ')[-1]))

    return data


top = int(sys.argv[1])

top_dict = {1:0, 3:1, 5:2, 10:3, 20:4}

param = []
hit = []
for hd in [64, 128, 256, 512, 1024]:
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        try:
            name = './results/hit_h%s_lr%s_valid.txt' % (hd, lr)
            data = []
            with open(name, 'r') as file:
                for line in file:
                    elems = line.strip('\n').split()
                    data.append(float(elems[-1]))

            r = data[top_dict[top]]
            param.append('hd%s lr%s' % (hd, lr))
            hit.append(r)
        except:
            pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

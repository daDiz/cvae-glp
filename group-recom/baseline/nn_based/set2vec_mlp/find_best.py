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
for ed in [32, 64, 128, 256]:
    for hd in [128, 256, 512, 1024]:
        for lr in [0.001, 0.01, 0.1]:
            for ts in [1]:
                try:
                    name = './results/hit_valid_ed%s_hd%s_lr%s_ts%s.txt' % (ed, hd, lr, ts)
                    data = []
                    with open(name, 'r') as file:
                        for line in file:
                            elems = line.strip('\n').split()
                            data.append(float(elems[-1]))

                    r = data[top_dict[top]]

                    param.append('ed%s hd%s lr%s ts%s' % (ed, hd, lr, ts))
                    hit.append(r)
                except:
                    pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])

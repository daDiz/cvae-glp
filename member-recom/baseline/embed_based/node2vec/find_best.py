import sys
import numpy as np
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
for d in [32, 64, 128]:
    for p in [0.5,1.0,1.5]:
        for q in [0.5, 1.0, 1.5]:
            for m in ['sum','max','min']:
                try:
                    name = './results/hit_rate_valid_%s_d%s_p%s_q%s.txt' % (m,d,p,q)
                    r = read_file(name)[k]
                    param.append('m: %s d: %s p: %s q: %s' % (m, d, p, q))
                    hit.append(r)
                except:
                    pass

ind_best = np.argmax(hit)
print('hit@%d : %.3f' % (top, hit[ind_best]))
print(param[ind_best])


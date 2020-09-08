import sys

f = sys.argv[1]
topk_list = [1,3,5,10,20]

hit_rate = {}

for d in [32, 64, 128]:
    for p in [0.5, 1.0, 1.5]:
        for q in [0.5, 1.0, 1.5]:
            for m in ['mean', 'sum','max','min']:
                with open('./results/hit_group_%s_%s_d%s_p%s_q%s.txt' % (f, m, d, p, q), 'r') as file:
                    for line in file:
                        elems = line.strip('\n').split()
                        k = int(elems[2][:-1])
                        rate = float(elems[-1])
                        if k in hit_rate:
                            if hit_rate[k][4] < rate:
                                hit_rate[k] = (m, d, p, q, rate)
                        else:
                            hit_rate[k] = (m, d, p, q, rate)


for k in topk_list:
    print('hit at %d: %.3f %s d%s p%s q%s' % (k, hit_rate[k][-1], hit_rate[k][0], hit_rate[k][1], hit_rate[k][2], hit_rate[k][3]))


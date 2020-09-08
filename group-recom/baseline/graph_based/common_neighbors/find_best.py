import sys

f = sys.argv[1]
topk_list = [1,3,5,10,20]

hit_rate = {}

for m in ['mean', 'sum','max','min']:
    with open('./results/hit_group_%s_%s.txt' % (f, m), 'r') as file:
        for line in file:
            elems = line.strip('\n').split()
            k = int(elems[2][:-1])
            rate = float(elems[-1])
            if k in hit_rate:
                if hit_rate[k][1] < rate:
                    hit_rate[k] = (m, rate)
            else:
                hit_rate[k] = (m, rate)


for k in topk_list:
    print('hit at %d: %.3f %s' % (k, hit_rate[k][1], hit_rate[k][0]))


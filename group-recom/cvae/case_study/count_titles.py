from collections import Counter
import numpy as np
#import matpotlib.pyplot as plt


#target = 'Employee'
#target = 'President'
#target = 'Vice President'
target = 'CEO'

all_roles = []

data = []
with open('example_test.txt', 'r') as file:
    for line in file:
        elems = line.strip('\n').split(',')
        A = elems[0].split('/')[1]
        all_roles.append(A)
        if A == target:
            for B in elems[1:]:
                B_ = B.split('/')[1]
                data.append((A,B_))

role_count = Counter(all_roles)
#role_n = np.sum([role_count[k] for k in role_count])
role_pct = [(k,role_count[k]) for k in role_count]
print(sorted(role_pct, key=lambda (x,y):-y))

title_count = Counter(data)

total = np.sum([title_count[k] for k in title_count])

print(total)
#print(title_count)

pair_percent = [(k, title_count[k]*1./total) for k in title_count]

pair_sorted = sorted(pair_percent, key=lambda (x,y): -y)

print(pair_sorted)

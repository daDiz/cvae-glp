import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('test_sample.pickle', 'rb') as file:
    pb = pickle.load(file)


#thres = list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1,1.1,0.1))
thres = np.arange(0.01, 1.01, 0.01)


count = []
for t in thres:
    n = []
    for k in pb:
        p = pb[k]
        n.append(np.sum(p > t) * 1. / len(p))

    count.append(100*np.mean(n))

#print(thres)
#print(count)
plt.plot(thres, count)
plt.xticks(np.arange(0.1,1.1,0.1))

plt.xlabel('probability threshold')
plt.ylabel('% percent')
#plt.show()
plt.savefig('elbow_thres.pdf')


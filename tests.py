import numpy as np 

l = np.load('static_force_4_nodes_at_1_in_y.npy')
d = np.arange(10)
f = [0,1]

k = list(set(d) - set(f))
s = np.ix_(k, k)
print(d[1::2])
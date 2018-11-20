import matplotlib.pyplot as plt
import numpy as np

method = 'rq'
dataset = 'netflix'

for method, marker, linestyle in [('rq', '^', '-.'), ('pq', '*', ':')]:
    x = np.genfromtxt(fname='{}/{}.log'.format(dataset, method), delimiter=',', skip_header=1)
    plt.plot(x[:, 1], 'blue', label=method+'_mse_errors', linestyle=linestyle, marker=marker)
    plt.plot(x[:, 2], 'red', label=method+'_norm_errors', linestyle=linestyle, marker=marker)
    plt.plot(x[:, 3], 'gray', label=method+'_angular_errors', linestyle=linestyle, marker=marker)

plt.legend(loc='upper right')
plt.show()
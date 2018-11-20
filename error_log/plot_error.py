import matplotlib.pyplot as plt
import numpy as np

method = 'pq'
dataset = 'netflix'

x = np.genfromtxt(fname='{}/{}.log'.format(dataset, method), delimiter=',', skip_header=1)
plt.plot(x[:, 1], 'blue', label='mse_errors', linestyle='-.', marker='^')
plt.plot(x[:, 2], 'red', label='norm_errors', linestyle='-.', marker='^')
plt.plot(x[:, 3], 'gray', label='angular_errors', linestyle='-.', marker='^')
plt.show()
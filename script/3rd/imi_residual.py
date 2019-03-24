import numpy as np
import matplotlib.pyplot as plt


# load data

dataset = np.genfromtxt('{}/imi_pq_error.sh'.format('sift1m'), delimiter=',')[:8]

fontsize=44
ticksize=36
plt.style.use('seaborn-white')

draw_error = True

plt.plot(dataset[:, 0], dataset[:, 1],
         label='Residual Encoding', color='red', linestyle='-', marker='s',
         markersize=6, linewidth=3)
plt.plot(dataset[:, 0], dataset[:, 2],
         label='Non-Residual Encoding', color='black', linestyle=':', marker='+',
         markersize=6, linewidth=3)

plt.yticks(fontsize=ticksize)
plt.xticks(fontsize=ticksize)
plt.xlabel('# codebook', fontsize=fontsize)
plt.ylabel('Quantization Error', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=ticksize)


plt.show()
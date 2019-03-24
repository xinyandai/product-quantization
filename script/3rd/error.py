import numpy as np
import matplotlib.pyplot as plt


# load data
music100 = np.genfromtxt('{}/error.sh'.format('music100'), delimiter=',')
imagenet = np.genfromtxt('{}/error.sh'.format('imagenet'), delimiter=',')

fontsize=44
ticksize=36
plt.style.use('seaborn-white')

draw_error = True

if draw_error:
    plt.plot(music100[:, 0] + 1, music100[:, 1] / 0.7727891, label='Music100', color='black', linestyle=':', marker='+',
             markersize=6, linewidth=3)
    plt.plot(imagenet[:, 0] + 1, imagenet[:, 1] / 0.29722556, label='ImageNet', color='red', linestyle='-', marker='s',
             markersize=6, linewidth=3)

    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.xlabel('# codebook', fontsize=fontsize)
    plt.ylabel('Quantization Error', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=ticksize)
else:
    plt.plot(music100[:, 0] + 1, music100[:, 2] / 0.7727891, label='Music100', color='black', linestyle=':', marker='+', markersize=6, linewidth=3)
    plt.plot(imagenet[:, 0] + 1, imagenet[:, 2] / 0.29722556, label='ImageNet', color='red', linestyle='-', marker='s', markersize=6, linewidth=3)

    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.xlabel('# codebook', fontsize=fontsize)
    plt.ylabel('Codebook Energy', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=ticksize)

plt.show()
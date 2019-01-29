import matplotlib.pyplot as plt
import numpy as np

dataset = 'yahoomusic'
start_lines = 0
end_lines = 17

fontsize=44
ticksize=36

plot_angular = False
# plot_angular = True
methods = [
        ('pq', 's', 'black', '-'),
        ('opq', '+', 'gray', '-.'),
        ('aq', 'o', 'red', '-'),
        ('rq', '*', 'blue', ':'),
    ]
markersize = 20
linewidth = 5
if plot_angular:
    for method, marker, color, linestyle in methods:
        x = np.genfromtxt(fname='{}/{}.log'.format(dataset, method), delimiter=',', skip_header=1)
        plt.plot(
            x[start_lines:end_lines, 0], x[start_lines:end_lines, 4], color,
            label=method.upper(), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.xlabel('# codebook', fontsize=fontsize)
    plt.ylabel('Angular Error', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

else:
    for method, marker, color, linestyle in methods:
        x = np.genfromtxt(fname='{}/{}.log'.format(dataset, method), delimiter=',', skip_header=1)
        plt.plot(
            x[start_lines:end_lines, 0], x[start_lines:end_lines, 2], color,
            label=method.upper(), linestyle=linestyle, marker=marker, markersize=markersize, linewidth=linewidth)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.xlabel('# codebook', fontsize=fontsize)
    plt.ylabel('Norm Error', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)


plt.subplots_adjust(
    top=0.97, bottom=0.148, left=0.068, right=0.99, hspace=0.2, wspace=0.163)

plt.show()
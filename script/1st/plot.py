import numpy as np
import matplotlib.pyplot as plt
import re

# data_set = 'tinygist10million'
# data_set = 'netflix'
# data_set = 'yahoomusic'
# data_set = 'sift1m'
data_set = 'imagenet'
# data_set = 'movielens'

top_k = 20
codebook = 8
Ks = 256


expected_avg_items = 0
overall_query_time = 1
avg_recall = 2
avg_precision = 3
avg_error_ratio = 4
actual_avg_items = 0



def read_csv(method):
    return np.genfromtxt("{}.sh".format(method), delimiter=',')


def plot_one(method, color, x, y, linestyle="-", marker='d'):
    data = read_csv(method)

    x = np.array(data[:, x])
    x = np.log(x)
    y = np.array(data[:, y])
    plt.plot(x, y, color, label=method, linestyle=linestyle, marker=marker)



def plot_(x_label, y_label, x, y):
    plt.title('%s - %s on %s of top%d with codebook-%d' % (y_label, x_label, data_set, top_k, codebook))
    plt.xlabel(x_label)
    plt.ylabel(y_label)


    plot_one('rq',        'blue', x, y, '--', '^')
    plot_one('opq',        'black', x, y, '--', '^')
    plot_one('orq_2layer', 'red',   x, y, '-.', '+')
    plot_one('orq_4layer', 'gray',  x, y, '-',  's')

    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    plot_('probe items', 'recall', expected_avg_items, avg_recall)
    # plot_('query time', 'recall', overall_query_time, avg_recall)
    # plot_('recall', 'precision', avg_recall, avg_precision)
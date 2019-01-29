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
codebook = 4
Ks = 256

fontsize=44
ticksize=36

expected_avg_items = 0
overall_query_time = 1
avg_recall = 2
avg_precision = 3
avg_error_ratio = 4
actual_avg_items = 0


def get_csv_file(method):
    return "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(method, color, x, y, linestyle="-", marker='d'):
    try:
        data = read_csv(get_csv_file(method))

        x = np.array(data[:10, x])
        # x = np.log(x)
        y = np.array(data[:10, y])
        method_name = method.replace("norm_rq", "NE-RQ")
        method_name = method_name.replace("irregular_norm_3_8_20", "Irregular NE-RQ")
        plt.plot(x, y, color, label=method_name, linestyle=linestyle, marker=marker, markersize=12, linewidth=3)
    except Exception as e:
        print(e)


def plot_(x, y):

    plt.xlabel('# Probe Items', fontsize=fontsize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plot_one('irregular_norm_3_8_20', 'red', x, y, ':', 's')
    plot_one('norm_rq',  'black',    x, y, '-', 'o')

    plt.legend(loc='lower right', fontsize=ticksize)
    plt.show()


if __name__ == '__main__':
    plot_(expected_avg_items, avg_recall)
    # plot_('query time', 'recall', overall_query_time, avg_recall)
    # plot_('recall', 'precision', avg_recall, avg_precision)

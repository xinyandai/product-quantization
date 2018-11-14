import numpy as np
import matplotlib.pyplot as plt
import re

top_k = 20
code_length = 32
# data_set = 'tinygist10million'
data_set = 'netflix'
# data_set = 'yahoomusic'
# data_set = 'imagenet'
# data_set = 'movielens'

codebook = 2
Ks = 256


expected_avg_items = 0
overall_query_time = 1
avg_recall = 2
avg_precision = 3
avg_error_ratio = 4
actual_avg_items = 0


def get_csv_file(method):
    return "%s/%d_%d_%s.sh.log" % (data_set, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(method, color, x, y, linestyle="-", marker='d'):
    try:
        data = read_csv(get_csv_file(method))

        x = np.array(data[1:, x])
        y = np.array(data[1:, y])
        data_list = "\n  ".join(["(%s, %s)" % (x[i], y[i]) for i in range(len(x))])
        method_name = method.replace("_", "\\_")
        print ('\\addplot \n coordinates \n {\n%s\n};\n\\addlegendentry{%s}' % (data_list, method_name))
        plt.plot(x, y, color, label=method, linestyle=linestyle, marker=marker)
    except Exception as e:
        print(e)


def plot_(x_label, y_label, x, y):
    plt.title('%s - %s on %s of top%d with code-%d' % (y_label, x_label, data_set, top_k, code_length))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # plot_one('norm_range',                  'gray',     x, y, '-',  'o')
    #
    plot_one('aq',      'magenta', x, y, '--', '+')
    # plot_one('norm_aq', 'magenta', x, y, '-.', '<')
    print('------------aq ----------------------------------------------')

    plot_one('pq',      'blue',    x, y, '--', '+')
    plot_one('norm_pq', 'blue',    x, y, '-.', '<')
    print('------------pq-----------------------------------------------')

    plot_one('rq',       'red',    x, y, '-.', '+')
    plot_one('norm_rq',  'red',    x, y, '-.', '<')
    print('-----------rq-----------------------------------------------')

    plt.legend(loc='lower right')

    # plt.savefig("%s.png" % get_csv_file('all'))
    plt.show()


if __name__ == '__main__':
    plot_('probe items', 'recall', expected_avg_items, avg_recall)
    # plot_('query time', 'recall', overall_query_time, avg_recall)
    # plot_('recall', 'precision', avg_recall, avg_precision)

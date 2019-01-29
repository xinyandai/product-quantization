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


def get_csv_file(method):
    return "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(method, color, x, y, linestyle="-", marker='d'):
    try:
        data = read_csv(get_csv_file(method))

        x = np.array(data[2:10, x])
        # x = np.log(x)
        y = np.array(data[2:10, y])
        data_list = "\n  ".join(["(%s, %s)" % (x[i], y[i]) for i in range(len(x))])
        method_name = method.replace("_", "\\_")
        print ('\\addplot \n coordinates \n {\n%s\n};\n\\addlegendentry{%s}' % (data_list, method_name))
        plt.plot(x, y, color, label=method, linestyle=linestyle, marker=marker)
    except Exception as e:
        print(e)


def plot_(x_label, y_label, x, y):
    plt.title('%s - %s on %s of top%d with codebook-%d' % (y_label, x_label, data_set, top_k, codebook))
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # plot_one('norm_range',                  'gray',     x, y, '-',  'o')
    #
    # plot_one('aq',      'black', x, y, '--', '+')
    # plot_one('norm_aq', 'black', x, y, '-.', '<')
    # plot_one('apq', 'green', x, y, '-.', '+')
    # print('------------aq ----------------------------------------------')

    # plot_one('pq',      'blue',    x, y, '--', '+')
    # plot_one('norm_pq', 'blue',    x, y, '-.', '<')
    # print('------------pq-----------------------------------------------')

    # plot_one('rq',       'red',    x, y, '-.', '+')
    # plot_one('norm_rq',  'red',    x, y, '-.', '<')
    plot_one('norm_rq_kmeans',  'black',    x, y, '-.', '<')
    plot_one('norm_rq_partial_kmeans',  'red',    x, y, '-.', '<')
    # plot_one('norm_orq', 'green',    x, y, '-.', '<')
    print('-----------rq-----------------------------------------------')

    # plot_one('opq',      'gray', x, y, '-.', '+')
    # plot_one('norm_opq', 'gray', x, y, '-.', '<')
    print('-----------opq-----------------------------------------------')

    plt.legend(loc='lower right')

    # plt.savefig("%s.png" % get_csv_file('all'))
    plt.show()


if __name__ == '__main__':
    plot_('probe items', 'recall', expected_avg_items, avg_recall)
    # plot_('query time', 'recall', overall_query_time, avg_recall)
    # plot_('recall', 'precision', avg_recall, avg_precision)

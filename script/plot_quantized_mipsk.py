import matplotlib.pyplot as plt
import numpy as np

# data_set = 'tinygist10million'
# data_set = 'netflix'
# data_set = 'yahoomusic'
# data_set = 'imagenet'
# data_set = 'movielens'

# top_k = 20
# codebook = 8
# Ks = 256


expected_avg_items = 0
overall_query_time = 1
avg_recall = 2
avg_precision = 3
avg_error_ratio = 4
actual_avg_items = 0

# fontsize = 20
# ticksize = 18
fontsize=56
ticksize=48
plt.style.use('seaborn-white')


def get_csv_file(data_set, top_k, codebook, Ks, method):
    # if codebook == 16 and method == 'norm_rq':
    #     return  "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, 'norm_2_rq')
    return "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(data_set, top_k, codebook, Ks, method, method_name, color, x, y, linestyle="-", marker='d', lines=10):
    data = read_csv(get_csv_file(data_set, top_k, codebook, Ks, method))

    x = np.array(data[:lines, x])
    # x = np.log(x)
    y = np.array(data[:lines, y])
    method_name = method_name.replace("(", " (")
    # method_name = method_name.replace("rq", "RQ")
    plt.plot(x, y, color, label=method_name, linestyle=linestyle, marker=marker, markersize=12, linewidth=3)


def plot():
    x = expected_avg_items
    y = avg_recall
    top_k = 20
    codebook = 8
    Ks = 256
    data_set = 'sift100m'
    lines = 14
    def _plot_setting():
        plt.xlabel('# Probe Items', fontsize=fontsize)
        plt.ylabel('Recall', fontsize=fontsize)
        plt.yticks(fontsize=ticksize)
        plt.xticks(fontsize=ticksize)
        plt.legend(loc='lower right', fontsize=ticksize-5,
                   handlelength=1,
                   borderpad=-0.4,
                   columnspacing=0.3,
                   handletextpad=0.2)
        plt.show()

    plot_one(data_set, top_k, codebook, Ks, 'norm_rq', 'NE-RQ', 'red', x, y, '-', '+',lines=lines)
    plot_one(data_set, top_k, codebook, Ks, 'norm_opq', 'NE-OPQ', 'coral', x, y, '-', '+',lines=lines)
    plot_one(data_set, top_k, codebook, Ks, 'norm_pq', 'NE-PQ', 'pink', x, y, '-', '+',lines=lines)
    # plot_one(data_set, top_k, codebook, Ks, 'opq', 'OPQ', 'yellow', x, y, '-', '+',lines=lines)
    plot_one(data_set, top_k, codebook, Ks, 'quantized_mips', 'QUIP-cov(q)', 'black', x, y, ':', '+', lines=lines-1)
    _plot_setting()

plot()

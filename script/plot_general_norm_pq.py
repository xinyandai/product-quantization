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
fontsize=22
ticksize=18
plt.style.use('seaborn-white')


def get_csv_file(data_set, top_k, codebook, Ks, method):
    return "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(data_set, top_k, codebook, Ks, method, color, x, y, linestyle="-", marker='d', lines=10):
    try:
        data = read_csv(get_csv_file(data_set, top_k, codebook, Ks, method))

        x = np.array(data[1:lines, x])
        # x = np.log(x)
        y = np.array(data[1:lines, y])
        method_name = method.replace("norm_", "NE-")
        plt.plot(x, y, color, label=method_name.upper(), linestyle=linestyle, marker=marker, markersize=8, linewidth=1.5)
    except Exception as e:
        print(e)


def plot():
    x = expected_avg_items
    y = avg_recall
    top_k = 20
    codebook = 8
    Ks = 256

    def _plot_setting(data_set):
        plt.xlabel('# Probe Items', fontsize=fontsize)
        plt.ylabel('Recall', fontsize=fontsize)
        plt.yticks(fontsize=ticksize)

        plt.xticks(fontsize=ticksize)
        plt.text(1, 0.9, data_set, fontsize=ticksize, color='black')
        plt.ylim(0, 1)
        plt.legend(loc='lower right', fontsize=ticksize)
        # plt.legend(loc='lower right', fontsize=ticksize - 5,
        #            handlelength=1, borderpad=-0.2,
        #            columnspacing=0.3, handletextpad=0.2)

    for i, (data_set, lines) in enumerate([('netflix', 8), ('yahoomusic', 8), ('imagenet', 10), ('sift100m', 14)]):
        plt.subplot(2, 4, i+1)

        plot_one(data_set, top_k, codebook, Ks, 'norm_opq', 'red', x, y, '-', 'X', lines=lines)
        plot_one(data_set, top_k, codebook, Ks, 'norm_pq', 'red', x, y, '-', 's', lines=lines)

        plot_one(data_set, top_k, codebook, Ks, 'opq', 'gray', x, y, '--', 'X', lines=lines)
        plot_one(data_set, top_k, codebook, Ks, 'pq', 'gray', x, y, '--', 's', lines=lines)

        _plot_setting(data_set)


    for i, (data_set, lines) in enumerate([('netflix', 8), ('yahoomusic', 8), ('imagenet', 10), ('sift100m', 14)]):

        plt.subplot(2, 4, i+1+4)
        plot_one(data_set, top_k, codebook, Ks, 'norm_rq', 'red', x, y, '-', 'X', lines=lines)
        plot_one(data_set, top_k, codebook, Ks, 'norm_aq', 'red', x, y, '-', 's', lines=lines)
        plot_one(data_set, top_k, codebook, Ks, 'rq', 'gray', x, y, '--', 'X', lines=lines)
        plot_one(data_set, top_k, codebook, Ks, 'aq', 'gray', x, y, '--', 's', lines=lines)
        _plot_setting(data_set)

    plt.subplots_adjust(
        top=0.97, bottom=0.148, left=0.068, right=0.99, hspace=0.2, wspace=0.163)

    plt.show()


plot()

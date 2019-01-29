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

fontsize = 20
ticksize = 18

plt.style.use('seaborn-white')


def get_csv_file(data_set, top_k, codebook, Ks, method):
    # if codebook == 16 and method == 'norm_rq':
    #     return  "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, 'norm_2_rq')
    return "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(data_set, top_k, codebook, Ks, method, color, x, y, linestyle="-", marker='d', lines=10):
    try:
        data = read_csv(get_csv_file(data_set, top_k, codebook, Ks, method))

        x = np.array(data[:lines, x])
        # x = np.log(x)
        y = np.array(data[:lines, y])
        method_name = method.replace("norm_rq", "NE-RQ")
        method_name = method_name.replace("rq", "RQ")
        plt.plot(x, y, color, label=method_name, linestyle=linestyle, marker=marker)
    except Exception as e:
        print(e)


def plot():
    x = expected_avg_items
    y = avg_recall
    top_k = 20
    Ks = 256

    data_set = 'sift100m'
    lines = 12
    if False:
        for i, codebook in enumerate([4, 6, 12, 16]):
            plt.subplot(1, 4, i + 1)
            plot_one(data_set, top_k, codebook, Ks, 'norm_rq', 'red', x, y, '-', 's', lines=lines)
            plot_one(data_set, top_k, codebook, Ks, 'rq', 'black', x, y, ':', '+', lines=lines)

            plt.xlabel('# Probe Items', fontsize=fontsize)
            plt.ylabel('Recall', fontsize=fontsize)
            plt.yticks(fontsize=ticksize)

            plt.legend(loc='lower right', fontsize=ticksize)
    else:
        codebook = 8
        for i, top_k in enumerate([1, 5, 10, 50]):
            plt.subplot(1, 4, i+1)
            plot_one(data_set, top_k, codebook, Ks, 'norm_rq', 'red', x, y, '-', 's', lines=lines)
            plot_one(data_set, top_k, codebook, Ks, 'rq', 'black', x, y, ':', '+', lines=lines)

            plt.xlabel('# Probe Items', fontsize=fontsize)
            plt.ylabel('Recall', fontsize=fontsize)
            plt.yticks(fontsize=ticksize)

            plt.legend(loc='lower right', fontsize=ticksize)

    plt.subplots_adjust(
        top=0.97, bottom=0.148, left=0.068, right=0.99, hspace=0.2, wspace=0.163)
    plt.show()


plot()

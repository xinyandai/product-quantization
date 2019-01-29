import matplotlib.pyplot as plt
import matplotlib
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
fontsize=44
ticksize=36
plt.style.use('seaborn-white')


def get_csv_file(data_set, top_k, codebook, Ks, method):
    if method == 'rational_nr_shift_4' or method == 'simpleLSH':
        return "%s/%d/code%d/%s.sh" % (data_set, top_k, Ks, method)
    return "%s/%d/%d_%d_%s.sh.log" % (data_set, top_k, codebook, Ks, method)


def read_csv(file_name):
    return np.genfromtxt(fname=file_name, delimiter=',', skip_header=1)


def plot_one(data_set, top_k, codebook, Ks, method, color, x, y, linestyle="-", marker='d', lines=10):
    try:
        data = read_csv(get_csv_file(data_set, top_k, codebook, Ks, method))

        x = np.array(data[1:lines, x])
        # x = np.log(x)
        y = np.array(data[1:lines, y])
        method_name = method.replace("norm_rq", "NE-RQ")
        method_name = method_name.replace("rq", "RQ")
        method_name = method_name.replace("rational_nr_shift_4", "Norm-Range-{}".format(Ks))
        method_name = method_name.replace("simpleLSH", "Simple-LSH-{}".format(Ks))
        plt.plot(x, y, color, label=method_name, linestyle=linestyle, marker=marker, markersize=12, linewidth=3)
    except Exception as e:
        print(e)


def plot():
    x = expected_avg_items
    y = avg_recall
    top_k = 20
    codebook = 2
    Ks = 256

    ax = plt.gca()
    mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 \
        else '%1.1fK' % (x * 1e-3) if x >= 1e3 \
        else '%1.0f' % x if x > 0 \
        else '0 '
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    data_set = 'imagenet'
    lines = 10
    # hash_method = 'rational_nr_shift_4'
    hash_method = 'simpleLSH'

    plot_one(data_set, top_k, codebook, Ks, 'norm_rq', 'red', x, y, '-', 's', lines=lines)
    plot_one(data_set, top_k, codebook, Ks, 'rq', 'blue', x, y, '-', '+', lines=lines)
    plot_one(data_set, top_k, codebook, 64, hash_method, 'black', x, y, '-.', "s", lines=lines)
    plot_one(data_set, top_k, codebook, 16, hash_method, 'gray', x, y, ':', "+", lines=lines)

    plt.xlabel('# Probe Items', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='upper left', fontsize=32)

    plt.subplots_adjust(
        top=0.97, bottom=0.148, left=0.068, right=0.99, hspace=0.2, wspace=0.163)

    plt.show()


plot()

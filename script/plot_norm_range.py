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
fontsize=56
ticksize=48
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
        method_name = method_name.replace("norm_pq", "NE-PQ")
        method_name = method_name.replace("rq", "RQ")
        method_name = method_name.replace("quip", "QUIP")
        method_name = method_name.replace("rational_nr_shift_4", "Norm-Range-{}".format(Ks))
        method_name = method_name.replace("simpleLSH", "Simple-LSH-{}".format(Ks))
        plt.plot(x, y, color, label=method_name, linestyle=linestyle, marker=marker, markersize=12, linewidth=3)
    except Exception as e:
        print(e)


def plot_hsnw(dataset='imagenet'):
    if dataset == "imagenet" :
        items = [
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    131072,
                    262144,
                    524288,
                    1048576,
                    2097152,
                    4194304,
                 ]
        recalls = [

                    0.01105,
                    0.0212,
                    0.0394,
                    0.0692,
                    0.1188,
                    0.18935,
                    0.28445,
                    0.40385,
                    0.52495,
                    0.6482,
                    0.754,
                    0.83785,
                    0.8978,
                    0.9345,
                    0.9574,
                    0.97205,
                    0.9826,
                    0.98995,
                    0.99435,
                    0.99785,
                    0.99965,
                    1,
        ]
        times = [34, 36, 38, 42, 48, 57, 73, 100, 157, 268, 591, 987,
                 2090, 3998, 9204, 17604, 28013, 70221, 143104,
                 280740, 552200, 625062, ]
        hnsw = [765, 537, 0.4747, 10,
                         1173, 701, 0.5679, 20,
                         1542, 818, 0.6262, 30,
                         1886, 1017, 0.664, 40,
                         2211, 1204, 0.6923, 50,
                         2536, 1386, 0.717, 60,
                         2850, 1571, 0.7345, 70,
                         3147, 1743, 0.7476, 80,
                         3439, 1917, 0.7637, 90,
                         3718, 2074, 0.7737, 100,
                         4260, 2398, 0.7923, 120,
                         4790, 2707, 0.8052, 140,
                         5300, 3014, 0.8186, 160,
                         5798, 3318, 0.8301, 180,
                         6274, 3602, 0.8394, 200,
                         6737, 3877, 0.8457, 220,
                         7189, 4160, 0.852, 240,
                         7623, 4414, 0.8564, 260,
                         8056, 4665, 0.8614, 280,
                         8480, 4916, 0.8651, 300,
                         10497, 6145, 0.8794, 400,
                         12367, 7299, 0.8879, 500,
                         14110, 8416, 0.8939, 600,
                         15755, 9473, 0.8967, 700,
                         17328, 10516, 0.9013, 800,
                         18845, 11441, 0.904, 900,
                         32740, 20258, 0.9171, 2000,
                         51705, 32706, 0.9246, 4000,
                         66823, 34646, 0.9275, 6000,
                         79671, 42009, 0.9292, 8000,
                         90992, 48977, 0.931, 10000,
                         101211, 55533, 0.9321, 12000,
                         110520, 61264, 0.9324, 14000,
                         119106, 66670, 0.9333, 16000,
                         127074, 72401, 0.9341, 18000,
                         ]
    elif dataset == "yahoomusic":
        items = [
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
        ]
        recalls = [
                    0.0708,
                    0.1328,
                    0.23875,
                    0.395,
                    0.5934,
                    0.78605,
                    0.9131,
                    0.97015,
                    0.991,
                    0.99715,
                    0.99945,
                    0.9998,
                    1,
                    1,
                    1,
        ]
        hnsw = [
            520, 142, 0.8388, 10,
            752, 214, 0.9082, 20,
            956, 282, 0.941, 30,
            1138, 341, 0.9568, 40,
            1308, 398, 0.9649, 50,
            1471, 540, 0.9724, 60,
            1621, 560, 0.9764, 70,
            1765, 558, 0.9792, 80,
            1902, 608, 0.9812, 90,
            2038, 657, 0.9836, 100,
            2290, 752, 0.9872, 120,
            2527, 844, 0.9891, 140,
            2753, 932, 0.9904, 160,
            2969, 1015, 0.9921, 180,
            3173, 1209, 0.9929, 200,
            3372, 1197, 0.9936, 220,
            3567, 1253, 0.994, 240,
            3755, 1331, 0.9949, 260,
            3936, 1406, 0.9953, 280,
            4113, 1484, 0.9955, 300,
            4937, 1842, 0.9961, 400,
            5680, 2173, 0.9964, 500,
            6380, 2498, 0.9966, 600,
            7030, 2810, 0.9971, 700,
            7646, 3107, 0.9976, 800,
            8236, 3395, 0.9977, 900,
            13538, 6213, 0.9985, 2000,
            20732, 10592, 0.9986, 4000,
            26510, 14529, 0.9986, 6000,
            31524, 18174, 0.9987, 8000,
            35993, 21757, 0.9987, 10000,
            40070, 25199, 0.9987, 12000,
            43830, 28456, 0.9987, 14000,
            47342, 31626, 0.9987, 16000,
            50647, 34856, 0.9987, 18000,
        ]

        times = [
                    37,
                    38,
                    40,
                    43,
                    53,
                    65,
                    88,
                    137,
                    310,
                    585,
                    910,
                    1742,
                    3927,
                    7758,
                    15784,
        ]

    items = np.array(items)
    times = np.array(times)
    recalls = np.array(recalls)
    hnsw = np.array(hnsw).reshape((-1, 4))
    print(hnsw)
    # plt.plot(times / 1000, recalls, 'red', label="NE-RQ", linestyle='-', marker='s', markersize=12, linewidth=3)
    # plt.xlabel('Time (ms)', fontsize=fontsize)
    # plt.plot(hnsw[:, 1] / 1000, hnsw[:, 2], 'black', label="ip-NSW", linestyle='--', marker="*", markersize=12,
    #          linewidth=3)

    ax = plt.gca()
    mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 \
        else '%1.0fK' % (x * 1e-3) if x >= 1e3 \
        else '%1.0f' % x if x > 0 \
        else '0 '
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    plt.plot(items, recalls, 'red', label="NE-RQ", linestyle='-', marker='s', markersize=12, linewidth=3)
    plt.xlabel('# Inner Product Computation', fontsize=fontsize)
    plt.plot(hnsw[:, 0], hnsw[:, 2], 'black', label="ip-NSW", linestyle='--', marker="*", markersize=12,
             linewidth=3)

    plt.xticks(fontsize=ticksize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='lower right', fontsize=ticksize, handlelength=1, borderpad=-0.4)

    plt.subplots_adjust(
        top=0.97, bottom=0.148, left=0.068, right=0.99, hspace=0.2, wspace=0.163)

    plt.show()


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
    # hash_method = 'simpleLSH'

    plot_one(data_set, top_k, codebook, 64, 'simpleLSH', 'gray', x, y, ':', "+", lines=lines)
    plot_one(data_set, top_k, codebook, 64, 'rational_nr_shift_4', 'black', x, y, '-.', "s", lines=lines)
    plot_one(data_set, top_k, codebook, Ks, 'quip', 'blue', x, y, '-', '+', lines=lines)
    plot_one(data_set, top_k, codebook, Ks, 'norm_pq', 'red', x, y, '-', 's', lines=lines)

    plt.xlabel('# Probe Items', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.ylabel('Recall', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='upper left', fontsize=ticksize, handlelength=1, borderpad=-0.4)

    plt.subplots_adjust(
        top=0.97, bottom=0.148, left=0.068, right=0.99, hspace=0.2, wspace=0.163)

    plt.show()


plot()

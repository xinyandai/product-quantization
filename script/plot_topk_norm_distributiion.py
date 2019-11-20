import numpy as np
import matplotlib.pyplot as plt
from vecs_io import loader

# data_set = 'tinygist10million'
# data_set = 'netflix'
# data_set = 'yahoomusic'
# data_set = 'sift1m'
# data_set = 'imagenet'
# data_set = 'movielens'

top_k = 20
codebook = 8
Ks = 256

fontsize=44
ticksize=36

expected_avg_items = 0
overall_query_time = 1
avg_recall = 2
avg_precision = 3
avg_error_ratio = 4
actual_avg_items = 0


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def topk_distribution():
    topk = [0, 5, 15, 20, 25, 100]
    base = "/research/jcheng2/xinyan/liujie/gqr/data"
    # base = "/research/jcheng2/xinyan/data"
    # base = "/home/xinyan/program/data"
    dataset = "new_tiny5m"
    X = fvecs_read('%s/%s/%s_base.fvecs'
                   % (base, dataset, dataset))
    G = ivecs_read('%s/%s/10_%s_product_groundtruth.ivecs'
                   % (base, dataset, dataset))
    G = G[:, :10]
    top_k_set = np.unique(G)
    norms = np.linalg.norm(X, axis=1)
    arg_norms = np.argsort(- norms)
    def percent_number(_percent):
        return   _percent * len(X) / 100
    percents = [
        len(np.intersect1d(
            arg_norms[topk[i-1]* len(X) // 100 : topk[i]* len(X) // 100 ], top_k_set)) / len(top_k_set)
        for i in range(1, len(topk))
    ]
    print(percents)

    x = range(len(percents))
    plt.bar(x, percents,
            tick_label=['{}-{}%'.format(topk[i-1], topk[i]) for i in range(1, len(topk))],
            fc='darkgray')
    plt.xlabel('Norm Ranking', fontsize=fontsize)
    plt.ylabel('Percentage', fontsize=fontsize)

    plt.xticks(fontsize=ticksize - 8)
    plt.yticks(fontsize=ticksize)

    plt.savefig("topk_distribution_{}.pdf".format(dataset))
    plt.show()



def topk_distribution_with_scales():
    split = 20
    top = 5

    dataset = "yahoomusic"
    for scale in ["0.25"]:
        X = fvecs_read('/home/xinyan/program/data/%s/'
                       '%s_base.fvecs'
                       % (dataset, dataset))
        G = ivecs_read('/home/xinyan/program/data/%s/'
                       '10_%s_product_groundtruth.ivecs'
                       % (dataset, dataset))
        top_k_set = np.unique(G)
        norms = np.linalg.norm(X, axis=1)
        arg_norms = np.argsort(- norms)
        percents = [len(np.intersect1d(i, top_k_set)) / len(top_k_set) for i in np.array_split(arg_norms, split)]
        percents[top] = np.sum(percents[top:])
        percents = percents[:top + 1]
        print(percents)

        x = range(len(percents))
        plt.bar(x, percents,
                tick_label=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-100%'],
                fc='darkgray')
        plt.xlabel('Norm Ranking', fontsize=fontsize)
        plt.ylabel('Percentage', fontsize=fontsize)

        plt.xticks(fontsize=ticksize-8)
        plt.yticks(fontsize=ticksize)

        plt.show()



if __name__ == '__main__':
    topk_distribution()
    # plot_(expected_avg_items, avg_recall)
    # plot_('query time', 'recall', overall_query_time, avg_recall)
    # plot_('recall', 'precision', avg_recall, avg_precision)

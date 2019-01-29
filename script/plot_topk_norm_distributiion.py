import numpy as np
import matplotlib.pyplot as plt
from vecs_io import loader

# data_set = 'tinygist10million'
# data_set = 'netflix'
# data_set = 'yahoomusic'
# data_set = 'sift1m'
data_set = 'imagenet'
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


def topk_distribution():
    split = 20
    top = 5

    X, Q, G = loader('sift1m', 20, 'product', verbose=False)
    top_k_set = np.unique(G)
    norms = np.linalg.norm(X, axis=1)
    arg_norms = np.argsort(- norms)
    percents = [len(np.intersect1d(i, top_k_set)) / len(top_k_set) for i in np.array_split(arg_norms, split)]
    percents[top] = np.sum(percents[top:])
    percents = percents[:top + 1]
    print(percents)

    x = range(len(percents))
    plt.bar(x, percents, tick_label=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-100%'], fc='darkgray')
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

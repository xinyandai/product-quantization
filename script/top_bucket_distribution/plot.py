import numpy as np
import matplotlib.pyplot as plt


# data_set = 'sift1m'
data_set = 'imagenet'
method = 'opq_imi'
# method = 'ivfadc_4096'

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

    portion = np.genfromtxt('{}_{}_buckets_recall_distribution_top20.txt'.format(data_set, method))
    portion = portion[:100]
    intervals = np.array([0, 5, 15, 30, 50, 70, 100])
    index = np.array(intervals / 100.0 * len(portion), dtype=np.int)
    print(index)

    percents = [np.sum(
        portion[index[i-1]: index[i] ]) for i in range(1, len(intervals))]
    tick_label = ['{}-{}'.format(index[i-1], index[i]) for i in range(1, len(index))]

    # percents.append(1.0 - np.sum(percents))
    # tick_label.append('others')
    print(percents)

    # x = range(len(portion))
    plt.bar(range(len(percents)), percents, tick_label=tick_label, fc='darkgray')
    # plt.plot(range(len(portion)), portion)
    plt.xlabel('Bucket Ranking', fontsize=fontsize)
    plt.ylabel('Portion', fontsize=fontsize)

    plt.xticks(fontsize=ticksize-8)
    plt.yticks(fontsize=ticksize)

    plt.show()


if __name__ == '__main__':
    topk_distribution()
    # plot_(expected_avg_items, avg_recall)
    # plot_('query time', 'recall', overall_query_time, avg_recall)
    # plot_('recall', 'precision', avg_recall, avg_precision)
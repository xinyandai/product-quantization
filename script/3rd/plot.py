import numpy as np
import matplotlib.pyplot as plt

codebook = 4
dataset = 'sift1m'

probed_items = [2 ** i for i in [4, 8, 10, 12, 14, 16]]
topKs = [1, 10, 20, 50, 100, 1000]
axis_ranks = np.log2([2 ** i for i in range(18)])

# load data
rq = np.genfromtxt('{}/c{}/rq.sh'.format(dataset, codebook), delimiter=',')
pq = np.genfromtxt('{}/c{}/pq.sh'.format(dataset, codebook), delimiter=',')
opq = np.genfromtxt('{}/c{}/opq.sh'.format(dataset, codebook), delimiter=',')
irregular_rq = np.genfromtxt('{}/c{}/iregular_rq.sh'.format(dataset, codebook), delimiter=',')
irregular_pq = np.genfromtxt('{}/c{}/iregular_pq.sh'.format(dataset, codebook), delimiter=',')

# reshape to {probe items by top-k by recalls@R}
rq, pq, irregular_rq, irregular_pq, opq = [
    recalls[:, 2:-1].reshape(len(probed_items), len(topKs) + 1, len(axis_ranks))[:,1:,:]
    for recalls in [rq, pq, irregular_rq, irregular_pq, opq]
]

# for r, probe_size in enumerate(probed_items):
#     for k, topK in enumerate(topKs):
#
#         plt.title('probe {} Items on top  K {}'.format(probe_size, topK))
#         plt.xlabel('re-ranked items')
#         plt.ylabel('recall')
#
#         # plt.plot(axis_ranks, rq[r, k, :], label='RQ', color='black')
#         plt.plot(axis_ranks, pq[r, k, :], label='PQ', color='blue')
#         # plt.plot(axis_ranks, opq[r, k, :], label='OPQ', color='gray')
#         # plt.plot(axis_ranks, irregular_rq[r, k, :], label='IrregularRQ', color='red')
#         plt.plot(axis_ranks, irregular_pq[r, k, :], label='IrregularPQ', color='pink')
#         plt.legend(loc='lower right')
#
#         plt.savefig("{}/c{}/pic/{}_{}.png".format(dataset, codebook, probe_size, topK))
#         plt.show()

fontsize=44
ticksize=36
plt.style.use('seaborn-white')
top_index = 2 # top20
probe_index = 5 # 65536
# plt.title('probe {} Items on top  K {}'.format(probed_items[probe_index], topKs[top_index]))

# plt.plot(axis_ranks, rq[probe_index, top_index, :], label='Regular Computation', color='black', linestyle=':', marker='+', markersize=12, linewidth=3)
# plt.plot(axis_ranks, pq[probe_index, top_index, :], label='PQ', color='blue', linestyle='-.', marker='^')
plt.plot(axis_ranks, pq[probe_index, top_index, :], label='Regular Computation(PQ)', color='blue', linestyle='-.', marker='^')
plt.plot(axis_ranks, irregular_pq[probe_index, top_index, :], label='Irregular Computation(PQ)', color='red', linestyle='-', marker='s')
# plt.plot(axis_ranks, opq[probe_index, top_index, :], label='opq', color='gray', linestyle='-.', marker='*')
# plt.plot(axis_ranks, irregular_rq[probe_index, top_index, :], label='Irregular Computation', color='red', linestyle='-', marker='s', markersize=12, linewidth=3)

plt.yticks(fontsize=ticksize)
plt.xticks(fontsize=ticksize)
plt.xlabel('Log2(T)', fontsize=fontsize)
plt.ylabel('Recall', fontsize=fontsize)
plt.legend(loc='lower right', fontsize=ticksize)

plt.show()
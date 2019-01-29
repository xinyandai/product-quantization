import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from vecs_io import loader


fontsize=44
ticksize=36


def plot():
    for dataset in ['sift1m']:
        X, _, _ = loader(dataset, ground_metric='product', folder='../data/')
        norms = np.linalg.norm(X, axis=1)
        norms[:] = norms[:] / np.max(norms)

        ax =plt.gca()
        mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 \
            else '%1.0fK' % (x * 1e-3) if x >= 1e3 \
            else '%1.0f' % x if x > 0 \
            else '0 '
        mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
        ax.yaxis.set_major_formatter(mkformatter)

        plt.hist(np.tile(norms, 100), bins=100, color='dimgray')
        plt.xlabel('Norm', fontsize=fontsize)
        plt.xticks(fontsize=ticksize)
        plt.ylabel('Frequency', fontsize=fontsize)
        plt.yticks(fontsize=ticksize)

        plt.show()

plot()


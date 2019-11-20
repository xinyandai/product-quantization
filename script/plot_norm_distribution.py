import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.io

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')



fontsize=44
ticksize=36

def plot(dataset="new_tiny5m", normalize=True):
    scale = ""
    X = fvecs_read('/research/jcheng2/xinyan/liujie/gqr/data/%s/%s_%sbase.fvecs' % (dataset, dataset, scale))
    # X = fvecs_read('/research/jcheng2/xinyan/data/%s/%s_%sbase.fvecs' % (dataset, dataset, scale))
    # X = fvecs_read('/home/xinyan/program/data/%s/%s_base.fvecs' % (dataset, dataset))
    norms = np.linalg.norm(X, axis=1)
    norms = np.sort(norms)
    mean = np.mean(norms)
    p = norms[int(0.95 * len(norms))]
    print(mean, p, p/mean)

    if normalize:
        norms[:] = norms[:] / np.max(norms)

    ax = plt.gca()
    mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 \
        else '%1.0fK' % (x * 1e-3) if x >= 1e3 \
        else '%1.0f' % x if x > 0 \
        else '0 '
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.yaxis.set_major_formatter(mkformatter)
    print(np.count_nonzero(norms>8))
    plt.hist(norms, bins=50, color='dimgray')
    plt.xlabel('Norm', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.yticks(fontsize=ticksize)

    # plt.savefig(
    #     'norm_distribution_{}_{}.pdf'.format(
    #         dataset, "normalized" if normalize else ""
    #     )
    # )
    plt.show()
    plt.clf()


plot()

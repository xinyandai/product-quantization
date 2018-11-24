from vecs_io import *
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ
from numpy.linalg import norm as l2norm


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def execute(args, X, train_size=100000):
    verbose = False
    if args.quantizer in ['PQ'.lower(), 'RQ'.lower()]:
        pqs = [PQ(M=args.num_codebook, Ks=args.Ks, verbose=verbose) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs, verbose=verbose)
    elif args.quantizer in ['OPQ'.lower()]:
        pqs = [OPQ(M=args.num_codebook, Ks=args.Ks, verbose=verbose) for _ in range(args.layer)]
        quantizer = ResidualPQ(pqs=pqs, verbose=verbose)
    elif args.quantizer == 'AQ'.lower():
        quantizer = AQ(M=args.num_codebook, Ks=args.Ks, verbose=verbose)
    else:
        assert False
    if args.sup_quantizer == 'NormPQ'.lower():
        quantizer = NormPQ(args.norm_centroid, quantizer, true_norm=args.true_norm, verbose=verbose)

    quantizer.fit(X[:train_size], iter=20)
    compressed = quantizer.compress(X)

    mse_errors = [
        l2norm(X[i] - compressed[i])
        / l2norm(X[i])
        for i in range(len(X))
    ]
    norm_errors = [
        np.abs(
            l2norm(compressed[i]) - l2norm(X[i])
        )
        / l2norm(X[i])
        for i in range(len(X))
    ]
    angular_errors = [
        l2norm(
            X[i] / l2norm(X[i]) - compressed[i] / l2norm(compressed[i])
        )
        for i in range(len(X))
    ]

    return np.mean(mse_errors), np.mean(norm_errors), np.mean(angular_errors)


if __name__ == '__main__':

    import sys
    args = DotDict()
    args.dataset = sys.argv[1]
    args.quantizer = sys.argv[2].lower()
    args.Ks = 256

    X = fvecs_read('../data/%s/%s_base.fvecs' % (args.dataset, args.dataset))

    args.layer = 1
    args.num_codebook = 1
    print('codebook, mse_errors, norm_errors, angular_errors')
    for i in range(16):
        if args.quantizer in ['PQ'.lower(), 'OPQ'.lower(), "AQ".lower()]:
            args.num_codebook = i + 1
        elif args.quantizer == 'RQ'.lower():
            args.layer = i + 1
        else:
            assert False, 'no designated method, (O)PQ or RQ'
        mse_error, norm_error, angular_error = execute(args, X)
        print('{}, {}, {}, {}'.format(i, mse_error, norm_error, angular_error))

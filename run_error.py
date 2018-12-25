from vecs_io import *
from pq_residual import *
from sorter import *
from transformer import *
from aq import AQ
from opq import OPQ
from numpy.linalg import norm as l2norm
from scipy import spatial


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def execute(args, X, Q, G, train_size=100000):
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
    topk_angular_err = [
        [
            np.abs(
                (spatial.distance.cosine(q, X[x_i]) - spatial.distance.cosine(q, compressed[x_i]))
                / spatial.distance.cosine(q, X[x_i])
            )
            for x_i in G[q_i]
        ]
        for q_i, q in enumerate(Q)
    ]
    return np.mean(mse_errors), np.mean(norm_errors), \
           np.mean(angular_errors), np.mean(topk_angular_err)


if __name__ == '__main__':

    import sys
    args = DotDict()
    args.dataset = sys.argv[1]
    args.quantizer = sys.argv[2].lower()
    args.Ks = 256

    X, Q, G = loader(args.dataset, 50, 'product')

    args.layer = 1
    args.num_codebook = 1
    print('codebook, mse_errors, norm_errors, angular_errors, topk_angular_error')
    for i in range(16):
        if args.quantizer in ['PQ'.lower(), 'OPQ'.lower(), "AQ".lower()]:
            args.num_codebook = i + 1
        elif args.quantizer == 'RQ'.lower():
            args.layer = i + 1
        else:
            assert False, 'no designated method, (O)PQ or RQ'
        mse_error, norm_error, angular_error, topk_angular_error = execute(args, X, Q, G)
        print('{}, {}, {}, {}, {}'.format(
            i, mse_error, norm_error, angular_error, topk_angular_error))

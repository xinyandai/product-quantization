from vecs_io import *
from pq_residual import *
from aq import AQ
from xpq import XPQ
from run import execute


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    topk = int(sys.argv[2])
    codebook = int(sys.argv[3])
    Ks = int(sys.argv[4])

    X, Q, G = loader(dataset, topk, 'product')
    # pq, rq, or component of norm-pq
    quantizer = XPQ([AQ(codebook//2, Ks) for _ in range(2)])
    quantizer = NormPQ(n_percentile=256, quantize=quantizer)
    execute(quantizer, X, Q, G, 'product')

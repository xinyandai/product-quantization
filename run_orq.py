from vecs_io import *
from pq_residual import *
from opq import OPQ
from run import execute


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    topk = int(sys.argv[2])
    layer = int(sys.argv[3])
    Ks = int(sys.argv[4])

    X, Q, G = loader(dataset, topk, 'product')
    # pq, rq, or component of norm-pq
    quantizer = NormPQ(
        n_percentile=Ks,
        quantize=OPQ(M=1, Ks=Ks, layer=layer)
    )
    execute(quantizer, X, Q, G, 'product')

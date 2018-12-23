from vecs_io import *
from pq_residual import *
from sorter import *
from run import execute


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1]
    topk = int(sys.argv[2])
    codebook = int(sys.argv[3])
    Ks = int(sys.argv[4])

    X, Q, G = loader(dataset, topk, 'product')
    # pq, rq, or component of norm-pq
    pqs = [PQ(M=1, Ks=Ks) for _ in range(3)]
    quantizer = ResidualPQ(pqs=pqs)
    quantizer = NormPQ(n_percentile=codebook, quantize=quantizer)
    execute(quantizer, X, Q, G, 'product')

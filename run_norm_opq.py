from vecs_io import *
from run_pq import execute
from opq import OPQ
from pq_norm import NormPQ

if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256

    X, T, Q, G = loader(dataset, topk, 'product', folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = OPQ(M=codebook-1, Ks=Ks)
    quantizer = NormPQ(n_percentile=Ks, quantize=quantizer)
    execute(quantizer, X, T, Q, G, 'product')

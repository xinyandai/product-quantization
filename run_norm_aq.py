from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from aq import AQ

if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256

    X, T, Q, G = loader(dataset, topk, 'product', folder='data/')
    # pq, rq, or component of norm-pq
    pqs = [PQ(M=1, Ks=Ks) for _ in range(codebook)]
    quantizer = AQ(M=codebook-1, Ks=Ks)
    quantizer = NormPQ(n_percentile=Ks, quantize=quantizer)
    execute(quantizer,  X, T, Q, G, 'product')

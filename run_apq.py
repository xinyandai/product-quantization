from vecs_io import *
from aq import AQ
from xpq import XPQ
from run import execute


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 8
    Ks = 256

    X, T, Q, G = loader(dataset, topk, 'product', folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = XPQ([AQ(codebook//2, Ks) for _ in range(2)])
    execute(quantizer, X, T, Q, G, 'product')

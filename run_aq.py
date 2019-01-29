from sorter import *
from run_pq import execute
from vecs_io import loader
from aq import AQ

if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256

    X, T, Q, G = loader(dataset, topk, 'product', folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = AQ(M=codebook, Ks=Ks)
    execute(quantizer, X, T, Q, G, 'product')

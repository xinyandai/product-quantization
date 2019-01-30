from vecs_io import *
from run_pq import execute
from opq import OPQ
from pq_norm import NormPQ
from run_pq import parse_args

if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, topk, codebook, Ks, metric = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(dataset, topk, codebook, Ks, metric))

    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = OPQ(M=codebook-1, Ks=Ks)
    quantizer = NormPQ(n_percentile=Ks, quantize=quantizer)
    execute(quantizer, X, T, Q, G, metric)

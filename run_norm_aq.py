from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
from aq import AQ
from run_pq import parse_args

if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) == 6:
        dataset = sys.argv[1]
        topk = int(sys.argv[2])
        codebook = int(sys.argv[3])
        Ks = int(sys.argv[4])
        metric = sys.argv[5]
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}".format(dataset, topk, codebook, Ks, metric))

    X, T, Q, G = loader(dataset, topk, metric, folder='data/')
    # pq, rq, or component of norm-pq
    pqs = [PQ(M=1, Ks=Ks) for _ in range(codebook)]
    quantizer = AQ(M=codebook-1, Ks=Ks)
    quantizer = NormPQ(n_percentile=Ks, quantize=quantizer)
    execute(quantizer,  X, T, Q, G, metric)

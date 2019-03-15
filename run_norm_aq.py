from vecs_io import *
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
    args = parse_args(dataset, topk, codebook, Ks, metric)
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder='data/')
    quantizer = AQ(M=args.codebook-1, Ks=args.Ks)
    quantizer = NormPQ(n_percentile=args.Ks, quantize=quantizer)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        with open(args.save_dir + '/' + args.dataset + '_norm_aq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)

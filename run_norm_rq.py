from vecs_io import *
from pq_residual import *
from sorter import *
from run_pq import execute
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
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder='data/')
    quantizer = ResidualPQ(M=args.num_codebook-1, Ks=args.Ks)
    quantizer = NormPQ(n_percentile=args.Ks, quantize=quantizer)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        with open(args.save_dir + '/' + args.dataset + '_norm_opq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)

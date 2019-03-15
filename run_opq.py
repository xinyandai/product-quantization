from vecs_io import *
from run_pq import execute
from opq import OPQ
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
        args = parse_args(dataset, topk, codebook, Ks, metric)
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = OPQ(M=args.num_codebook, args.Ks=args.Ks)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        with open(args.save_dir + '/' + args.dataset + '_opq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)

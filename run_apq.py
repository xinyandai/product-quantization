from vecs_io import *
from aq import AQ
from pqx import PQX
from run_pq import execute
from run_pq import parse_args


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 8
    Ks = 256
    metric = 'product'

    # override default parameters with command line parameters
    import sys
    args = parse_args()
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder='data/')
    # pq, rq, or component of norm-pq
    quantizer = PQX([AQ(args.num_codebook//2, args.Ks) for _ in range(2)])
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        with open(args.save_dir + '/' + args.dataset + '_apq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)

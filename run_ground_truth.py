from vecs_io import *
from sorter import *


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--topk', type=int, help='required topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    args = parser.parse_args()
    return args.dataset, args.topk, args.metric


def topk(data_set, top_ks, ground_metric):
    folder = 'data/'
    folder_path = folder + data_set
    base_file = folder_path + '/%s_base.fvecs' % data_set
    query_file = folder_path + '/%s_query.fvecs' % data_set

    print("# loading the base data {}, \n".format(base_file))
    X = fvecs_read(base_file)
    print("# loading the queries data {}, \n".format(query_file))
    Q = fvecs_read(query_file)
    print("# sorting")
    knn = Sorter(compressed=X, Q=Q[0:1000], X=X, metric=ground_metric).topK

    for top_k in top_ks:
        ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                       (top_k, data_set, ground_metric)
        ivecs_writer(ground_truth, np.array(knn[:, :top_k]))


if __name__ == "__main__":
    metric = 'product'
    dataset = 'netflix'
    top_k = 1

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, top_k, metric = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, metric = {}"
          .format(dataset, topk, metric))

    topk(dataset, [top_k], metric)

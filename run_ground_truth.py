from vecs_io import *
from sorter import *


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
    topk('netflix', [20], 'product')

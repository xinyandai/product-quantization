import numpy as np

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def recall():
    pass

def plot(dataset):
    topk = 10
    # base = "/home/xinyan/program/data/"
    base = "/research/jcheng2/xinyan/data/"
    product_knn = ivecs_read('%s/%s/50_%s_knn.ivecs' % (base, dataset, dataset))

    # first_hop_type = "angular"
    first_hop_type = "product"
    first_hop = ivecs_read('%s/%s/1000_%s_%s_groundtruth.ivecs' % (base, dataset, dataset, first_hop_type))
    product_gt = ivecs_read('%s/%s/1000_%s_product_groundtruth.ivecs' % (base, dataset, dataset))

    first_hop = first_hop[:,   : 10]
    product_knn = product_knn[:, : topk]
    product_gt = product_gt[:,   : topk]

    recall = 0.0
    for first_hop_nn, gt  in zip(first_hop, product_gt):
        nns= product_knn[first_hop_nn, :]
        recall += len(np.intersect1d(gt, np.unique(nns))) / topk
    print(recall / len(first_hop))


plot("imagenet")
# product-product imagenet 0.6720999999999994
# product-product yahoomusic 0.9177999999999973

# angular-product imagenet 0.9721999999999994
# angular-product yahoomusic 0.8266999999999949

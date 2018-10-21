import numpy as np


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def ivecs_read(filename, c_contiguous=True):
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    iv = iv[:, 1:]
    if c_contiguous:
        iv = iv.copy()
    return iv


def loader(data_set='audio', top_k=20, ground_metric='euclid'):
    folder_path = '../data/%s' % data_set
    base_file = folder_path + '/%s_base.fvecs' % data_set
    query_file = folder_path + '/%s_query.fvecs' % data_set
    ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                   (top_k, data_set, ground_metric)

    print("load the base data {}, \nload the queries {}, \nload the ground truth {}".format(base_file, query_file,
                                                                                            ground_truth))
    X = fvecs_read(base_file)
    Q = fvecs_read(query_file)
    G = ivecs_read(ground_truth)
    return X, Q, G


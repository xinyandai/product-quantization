import numpy as np
import struct


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()


# we mem-map the biggest files to avoid having them in memory all at
# once
def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def bvecs_read(filename):
    return mmap_bvecs(fname=filename)


def fvecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def loader(data_set='audio', top_k=20, ground_metric='euclid', folder='../data/', data_type='fvecs'):
    """
    :param data_set: data set you wanna load , audio, sift1m, ..
    :param top_k: how many nearest neighbor in ground truth file
    :param ground_metric:
    :param folder:
    :return: X, T, Q, G
    """
    folder_path = folder + data_set
    base_file = folder_path + '/%s_base.%s' % (data_set, data_type)
    train_file = folder_path + '/%s_learn.%s' % (data_set, data_type)
    query_file = folder_path + '/%s_query.%s' % (data_set, data_type)
    ground_truth = folder_path + '/%s_%s_%s_groundtruth.ivecs' % \
                   (top_k, data_set, ground_metric)

    print("# load the base data {}, \n# load the queries {}, \n# load the ground truth {}".format(base_file, query_file,
                                                                                            ground_truth))
    if data_type == 'fvecs':
        X = fvecs_read(base_file)
        Q = fvecs_read(query_file)
        try:
            T = fvecs_read(train_file)
        except FileNotFoundError:
            T = None
    elif data_type == 'bvecs':
        X = bvecs_read(base_file).astype(np.float32)
        Q = bvecs_read(query_file).astype(np.float32)
        try:
            T = bvecs_read(train_file)
        except FileNotFoundError:
            T = None
    else:
        assert False
    try:
        G = ivecs_read(ground_truth)
    except FileNotFoundError:
        G = None
    return X, T, Q, G


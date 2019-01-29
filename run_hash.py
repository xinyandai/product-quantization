from vecs_io import *
from transformer import *
from hash import RandomProjection


from run_pq import execute


if __name__ == '__main__':

    top_k = 20
    data_set = 'netflix'
    metric = "angular"
    folder = 'data/'

    def raw():
        X, T, Q, G = loader(data_set, top_k, metric)
        X, Q = scale(X, Q)

        pq = RandomProjection(512)
        execute(pq,  X, T, Q, G, 'sign')

    raw()
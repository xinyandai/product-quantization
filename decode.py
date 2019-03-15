from sorter import *
from transformer import *
from vecs_io import loader
import pickle


def gen_chunk(X, chunk_size=1000000):
    for i in range(math.ceil(len(X) / chunk_size)):
        yield X[i * chunk_size : (i + 1) * chunk_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')

    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    parser.add_argument('--chunk_size', type=int, help='chunk size', default=1000000)

    args = parser.parse_args()

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '.pickle', 'rb') as f:
        quantizer = pickle.load(f)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_encoded', 'rb') as f:
        codes = np.fromfile(f, dtype=quantizer.code_dtype).reshape(-1, quantizer.num_codebooks)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_decoded', 'wb') as f:
        for codes_chunk in gen_chunk(codes, args.chunk_size):
            decoded = quantizer.decode(codes_chunk)
            decoded.tofile(f)

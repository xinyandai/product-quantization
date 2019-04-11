import numpy as np

class RPTree:
    """
    A random projection tree.
    """
    def __init__(self, max_size):
        assert max_size >= 1
        self.max_size = max_size
        self.bid = 0


    def _gen_projector(self, X, idx):
        return np.random.randn(X.shape[1], 1)

    def make_tree(self, X, idx, buckets, root):
        """
        Recursively build a random projection tree
        from X, starting at the root.
        """

        N = len(idx)
        if N <= self.max_size:
            buckets[root.bid, :] = idx
            root.bid += 1
            return

        self.projector = self._gen_projector(X, idx)

        projected_value = (X[idx] @ self.projector).reshape(-1)
        self.sorted_idx = np.argsort(projected_value)
        self.sorted_left_inx = self.sorted_idx[:N//2]
        self.sorted_right_inx = self.sorted_idx[N//2:]

        self.threshold = np.mean(projected_value[self.sorted_idx[N//2-1: N//2+1]])

        self.left = RPTree(self.max_size)
        self.right = RPTree(self.max_size)

        self.left.make_tree(X, idx[self.sorted_left_inx], buckets, root)
        self.right.make_tree(X, idx[self.sorted_right_inx], buckets, root)


if __name__ == '__main__':
    max_size = 16
    tree = RPTree(16)
    X = np.random.normal(size=(256 * 16, 300))
    buckets = np.zeros(shape=(256, 16), dtype=np.int)
    tree.make_tree(X, np.arange(len(X)), buckets, tree)
    print(buckets)
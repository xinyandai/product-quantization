from scipy.stats import ortho_group
import numpy as np


def zero_mean(X, Q):
    mean = X.mean(axis=0, keepdims=True)
    X = X - mean
    Q = Q - mean
    return X, Q


def random_rotate(X, Q):
    R = ortho_group.rvs(dim=len(X[0]))
    R = np.array(R, dtype=np.float32)
    X = R.dot(X.transpose()).transpose()
    Q = R.dot(Q.transpose()).transpose()
    return X, Q


def scale(X, Q):
    scale = np.max(np.linalg.norm(X, axis=1))
    X /= scale
    Q /= scale
    return X, Q


def e2m_transform(X, Q):
    X_plus = np.full((len(X), 1), fill_value=-0.5, dtype=np.float32)
    Q_plus = np.full((len(Q), 1), fill_value=-0.5, dtype=np.float32)

    X_plus[:, 0] = np.linalg.norm(X, axis=1) ** 2
    X = np.append(X, X_plus, axis=1)
    Q = np.append(Q, Q_plus, axis=1)
    return X, Q


def e2m_mahalanobis(X):
    X_plus = np.full((len(X), 1), fill_value=-0.5, dtype=np.float32)
    X = np.append(X, X_plus, axis=1)
    return np.dot(X.transpose(), X) / float(len(X))

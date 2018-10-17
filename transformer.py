from scipy.stats import ortho_group
import numpy as np


def e2m_transform(X, Q) :
    # mean = X.mean(axis=0, keepdims=True)
    # X = X - mean
    # Q = Q - mean

    scale = np.max(np.linalg.norm(X, axis=1))
    X /= scale
    Q /= scale

    X_plus = np.full((len(X), 2), fill_value=-0.5, dtype=np.float32)
    X_plus[:, 0] = np.linalg.norm(X, axis=1) ** 2
    X = np.append(X, X_plus, axis=1)
    X = X - X.mean(axis=0, keepdims=True)

    Q_plus = np.full((len(Q), 2), fill_value=-0.5, dtype=np.float32)
    Q_plus[:, 1] = np.linalg.norm(Q, axis=1) ** 2
    Q = np.append(Q, Q_plus, axis=1)

    R = ortho_group.rvs(dim=len(X[0]))
    R = np.array(R, dtype=np.float32)
    X = R.dot(X.transpose()).transpose()
    Q = R.dot(Q.transpose()).transpose()
    return X, Q

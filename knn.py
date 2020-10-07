"""
Pythonic implementation of the paper:
    Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
    Non-linear metric learning. In NIPS (pp. 2573-2581).
"""
__author__ = "Iago Suarez"
__email__ = "iago.suarez.canosa@alumnos.upm.es"
import numpy as np
from sklearn.neighbors._ball_tree import BallTree


def LSKnn2(Ni, KK, MM):
    Ni = Ni.T
    N = Ni.shape[1]
    Ni = Ni - MM + 1
    classes = np.unique(Ni)
    T = np.zeros((len(classes), N, KK))
    for i, c in enumerate(classes):
        for k in range(KK):
            T[i, :, k] = np.sum(Ni[0:(k + 1), :] == c, axis=0)

    yy = np.zeros((np.max(KK), N), dtype=int)
    for k in range(KK):
        # Select the maximum T among all the classes
        arr = T[:, :, k] + T[:, :, 0] * 0.01
        temp, yy[k, 0:N] = np.max(arr, axis=0), np.argmax(arr, axis=0)
        # yy will contain the labels
        yy[k, 0:N] = classes[yy[k, :]]
    yy = yy[0:KK, :]
    yy = yy + MM - 1
    return yy


def knn_error_score(L, x_train, y_train, x_test, y_test, k, tree_size=15):
    """
    Measures the training and testing errors of a kNN classifier implemented using BallTree.
    :param L: linear transformation
    :param x_train: training vectors (each column is an instance)
    :param y_train: training labels  (row vector!!)
    :param x_test: test vectors
    :param y_test: test labels
    :param k: number of nearest neighbors
    :return: training and testing error in k-NN problem.
    """
    assert y_train.ndim == 1, y_test.ndim == 1
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    assert isinstance(k, (int, np.int32, np.int64)) and k > 0

    if len(L) != 0:
        # L is the initial linear projection, for example PCa or LDA
        x_train = x_train @ L.T
        x_test = x_test @ L.T

    tree = BallTree(x_train, leaf_size=tree_size, metric='euclidean')

    MM = np.append(y_train, y_test).min()
    NTr, NTe = x_train.shape[0], x_test.shape[0]

    # Use the tree to compute the distance between the testing and training points
    # iTe: indices of the testing elements in the training set
    dists, iTe = tree.query(x_test, k=k, return_distance=True)

    # Labels of the testing elements in the training set
    lTe2 = LSKnn2(y_train[iTe], k, MM)
    # Compute the error for each k
    test_error = np.sum(lTe2 != np.repeat(y_test, k, axis=0), axis=1) / NTe

    # Use the tree to compute the distance between the training points
    dists, iTr = tree.query(x_train, k=k + 1, return_distance=True)
    iTr = iTr[:, 1:]
    lTr2 = LSKnn2(y_train[iTr], k, MM)
    training_error = np.sum(lTr2 != np.repeat(y_train, k, axis=0), axis=1) / NTr

    return float(training_error), float(test_error)

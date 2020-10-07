import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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


def knn_error_score(L, xTr, lTr, xTe, lTe, k, treesize=15):
    """

    :param L: linear transformation
    :param xTr: training vectors (each column is an instance)
    :param lTr: training labels  (row vector!!)
    :param xTe: test vectors
    :param lTe: test labels
    :param k: number of nearest neighbors
    :return: training and testing error in k-NN problem.
    """
    assert lTr.ndim == 1, lTe.ndim == 1
    assert xTr.shape[0] == len(lTr)
    assert xTe.shape[0] == len(lTe)
    assert isinstance(k, (int, np.int32, np.int64)) and k > 0

    if len(L) != 0:
        # L is the initial linear projection, for example PCa or LDA
        xTr = xTr @ L.T
        xTe = xTe @ L.T

    tree = BallTree(xTr, leaf_size=treesize, metric='euclidean')

    MM = np.append(lTr, lTe).min()
    NTr, NTe = xTr.shape[0], xTe.shape[0]

    # Use the tree to compute the distance between the testing and training points
    # iTe: indices of the testing elements in the training set
    dists, iTe = tree.query(xTe, k=k, return_distance=True)

    # Labels of the testing elements in the training set
    lTe2 = LSKnn2(lTr[iTe], k, MM)
    # Compute the error for each k
    test_error = np.sum(lTe2 != np.repeat(lTe, k, axis=0), axis=1) / NTe

    # Use the tree to compute the distance between the training points
    dists, iTr = tree.query(xTr, k=k + 1, return_distance=True)
    iTr = iTr[:, 1:]
    lTr2 = LSKnn2(lTr[iTr], k, MM)
    training_error = np.sum(lTr2 != np.repeat(lTr, k, axis=0), axis=1) / NTr

    return float(training_error), float(test_error)

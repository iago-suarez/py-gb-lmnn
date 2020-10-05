import numpy as np
from scipy.spatial.distance import cdist


def LSKnn2(Ni, KK, MM):
    """
    Return a list of classes that TODO
    :param Ni:
    :param KK:
    :param MM:
    :return:
    """

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


def knncl(L, xTr, lTr, xTe, lTe, KK, train=True, test=True, cosigndist=0, blocksize=700, **kwargs):
    """

    :param L: transformation matrix (learned by LMNN)
    :param xTr: training vectors (each column is an instance)
    :param lTr: training labels  (row vector!!)
    :param xTe: test vectors
    :param lTe: test labels
    :param KK: number of nearest neighbors
    :param train:
    :param test:
    :param cosigndist:
    :param blocksize:
    :param kwargs:
    :return:
    """
    assert xTr.ndim == 2 and lTr.ndim == 1 and xTr.shape[1] == len(lTr)
    assert isinstance(KK, int)
    MM = min(lTr.min(), lTe.min())
    outputK = KK
    Kn = KK
    # TODO Avoid xTr and xTe to be transposed
    if L is not None and len(L) > 0:
        D = L.shape[1]
        xTr = L @ xTr[:D]
        if xTe is not None and len(xTe) > 0:
            xTe = L @ xTe[:D]
    else:
        D = xTr.shape[0]

    # Get sizes
    NTr = xTr.shape[1]
    NTe = xTe.shape[1]
    Eval = np.zeros((2, KK))
    lTr2 = np.zeros((KK, NTr))
    lTe2 = np.zeros((KK, NTe))
    iTr = np.zeros((Kn, NTr), dtype=int)
    iTe = np.zeros((Kn, NTe), dtype=int)

    def mink(D, k, axis=0):
        if k == 1:
            i = np.argmin(D, axis=axis)[np.newaxis]
        else:
            i = np.argsort(D, axis=axis)[:k]
        return np.take_along_axis(D, i, axis=0), i

    if train:
        pass  # TODO
    if test:
        Dtr = cdist(xTe.T, xTr.T, 'sqeuclidean').T
        dist, nn = mink(Dtr, Kn)
        lTe2 = LSKnn2(lTr[nn], KK, MM)
        iTe = nn
        Eval[1, :] = np.sum(lTe2 != np.repeat(lTe, KK, axis=0), axis=1) / NTe
    if train and test:
        print("Progress: Train: {}, Test: {}".format(100 * Eval[0], 100 * Eval[1]))
    elif train and not test:
        print("Progress: Train: {}".format(100 * Eval[0]))
    elif not train and test:
        print("Progress: tres: {}".format(100 * Eval[1]))

    # create "Details" output
    Details = []
    if test:
        Details = [lTe2, iTe]
    if train:
        Details = [lTr2, iTr]

    # extract "Eval" output
    if train and test:
        Eval = Eval[:, outputK - 1]
    elif train and not test:
        Eval = Eval[0, outputK - 1]
    elif not train and test:
        Eval = Eval[1, outputK - 1]

    return Eval, Details

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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


def eval_knn(X_tr, y_tr, L, X_te, y_te, k=3):
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)
    assert L.shape[0] == X_tr.shape[1] and L.shape[0] == X_te.shape[1]
    kNN = KNeighborsClassifier(n_neighbors=k).fit(X_tr @ L, y_tr)
    return np.mean(kNN.predict(X_te @ L) != y_te)

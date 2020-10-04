# Iago Suarez implementation of Non-linear Metric Learning
import matplotlib.pyplot as plt
import numpy as np
from metric_learn import LMNN
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from gb_lmnn import gb_lmnn
from mtrees import MTree


def findknnmtreeomp(x, testx, k):
    iknn, dists = None, None
    return iknn, dists


def usemtreemexomp(xtest, xtrain, tree, k):
    return tree.findknn(xtrain, xtest, k)


def LSKnn2(Ni, KK, MM):
    """
    Return a list of classes that TODO
    :param Ni:
    :param KK:
    :param MM:
    :return:
    """

    # if(nargin<2)
    #  KK=1:2:3;
    # end;
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


def pca(X, whiten=False):
    """
    finds principal components of
    :param X:  dxn matrix (each column is a dx1 input vector)
    :param whiten: Whether to divide by the eigenvalues
    :return: (evects, evals)
        - evects  columns are principal components (leading from left->right)
        - evals   corresponding eigenvalues
    """
    d, N = X.shape
    X = X - X.mean(axis=1)[:, np.newaxis]
    cc = np.cov(X)
    cdd, cvv = np.linalg.eig(cc)
    ii = np.argsort(-cdd)
    evects = cvv[:, ii]
    evals = cdd[ii]
    if whiten:
        evects /= evals
    return evects, evals


def knnclassifytreeomp(L, xTr, lTr, xTe, lTe, KK, tree=None, treesize=15, train=True, **kwargs):
    """

    :param L: linear transformation
    :param xTr: training vectors (each column is an instance)
    :param lTr: training labels  (row vector!!)
    :param xTe: test vectors
    :param lTe: test labels
    :param KK: number of nearest neighbors
    :param kwargs: Other arguments
    :return: Dictionary: {'tree': tree; 'teesize': 15; 'train':1; }
        - tree',tree (precomputed,mtree)
        - 'teesize',15 (max number of elements in leaf)
        - 'train',1  (0 means no training error)
    """
    assert lTr.ndim == 1, lTe.ndim == 1
    assert xTr.shape[1] == len(lTr)
    assert xTe.shape[1] == len(lTe)

    if len(L) != 0:
        # L is the initial linear projection, for example PCa or LDA
        xTr = L @ xTr
        xTe = L @ xTe

    if tree is None:
        print('Building tree ...')
        tree = MTree.build(xTr, treesize)

    MM = np.append(lTr, lTe).min()
    Kn = np.max(KK)
    assert np.alltrue(KK > 0)

    NTr = xTr.shape[1]
    NTe = xTe.shape[1]

    if xTr.shape[0] != xTe.shape[0]:
        raise ValueError('PROBLEM: Please make sure that training inputs and test inputs have the same dimensions!'
                         'xTr.shape: {}, xTe.shape: {}'.format(xTr.shape, xTe.shape))

    if tree.jumpindex.max() != (NTr - 1):
        raise ValueError('PROBLEM: Tree does not seem to belong to training data! '
                         'Max index of tree: ' + str(kwargs['tree'].jumpindex.max()) +
                         ' Length of training data: ' + str(NTr))

    print("Evaluating...")
    Eval = np.zeros((2, KK))
    # Use the tree to compute the distance between the testing and training points
    # iTe: indices of the testing elements in the training set
    iTe, dists = usemtreemexomp(xTe, xTr, tree, Kn)
    assert iTe.ndim == 2 and iTe.shape == (1, NTe)
    assert dists.ndim == 2 and dists.shape == (1, NTe)

    # Labels of the testing elements in the training set
    lTe2 = LSKnn2(lTr[iTe], KK, MM)
    # Compute the error for each k
    Eval[1] = np.sum(lTe2 != np.repeat(lTe, KK, axis=0), axis=1) / NTe

    Details = {}
    if train:
        # Use the tree to compute the distance between the training points
        iTr, dists = usemtreemexomp(xTr, xTr, tree, Kn + 1)
        iTr = iTr[1:]
        lTr2 = LSKnn2(lTr[iTr], KK, MM)
        Eval[0, :] = np.sum(lTr2 != np.repeat(lTr, KK, axis=0), axis=1) / NTr

        Details['lTr2'] = lTr2
        Details['iTr'] = iTr
    else:
        Eval[0] = []

    Details['lTe2'] = lTe2
    Details['iTe'] = iTe
    # Eval = Eval[:, outputK]
    return Eval, Details, tree


def eval_knn(X_tr, y_tr, L, X_te, y_te, k=3):
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)
    assert L.shape[0] == X_tr.shape[1] and L.shape[0] == X_te.shape[1]
    kNN = KNeighborsClassifier(n_neighbors=k).fit(X_tr @ L, y_tr)
    return np.mean(kNN.predict(X_te @ L) != y_te)


# #####################################################################################3

def main():
    print("--> ************************************************************************************")
    print("--> *************************** Metric Learning Demo ***********************************")
    print("--> ************************************************************************************")

    # Load variables
    print("--> Loading data")
    _, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('data/segment.mat').values()
    yTr, yTe = yTr.flatten(), yTe.flatten()
    err, _, _ = knnclassifytreeomp([], xTr, yTr, xTe, yTe, 1)

    L0 = pca(xTr)
    err, _, _ = knnclassifytreeomp(L0[0:3], xTr, yTr, xTe, yTe, 1)
    print('\n')
    print('1-NN Error after PCA in 3d is : {}%'.format(100 * err[1]))

    print("--> Training pca...")
    L0 = pca(xTr, whiten=True)[0].T

    print("--> Training pca-lda...")
    pca_lda = Pipeline([
        ('pca', PCA(n_components=5, whiten=True)),
        ('lda', LinearDiscriminantAnalysis(n_components=3))
    ])

    pca_lda.fit(xTr.T, yTr.flatten())
    lda_xtr, lda_xte = pca_lda.transform(xTr.T), pca_lda.transform(xTe.T)

    print("--> Training lmnn...")
    lmnn = LMNN(init='pca', k=7, learn_rate=1e-6, verbose=False, n_components=3, max_iter=1000)
    lmnn.fit(xTr.T, yTr.flatten())

    # ################################ k-NN evaluation ###################################
    print("\n--> Evaluation:")
    k = 1
    te_err = eval_knn(xTr.T, yTr.flatten(), np.eye(len(xTr)), xTe.T, yTe.flatten(), k)
    print('--> 1-NN Error for raw (high dimensional) input is : {}%'.format(100 * te_err))

    te_err = eval_knn(xTr.T, yTr.flatten(), L0[0:3].T, xTe.T, yTe.flatten(), k)
    print('--> 1-NN Error for PCA is : {}%'.format(100 * te_err))

    te_err = eval_knn(lda_xtr, yTr.flatten(), np.eye(3), lda_xte, yTe.flatten(), k)
    print('--> 1-NN Error for PCA-LDA input is : {}%'.format(100 * te_err))

    te_err = eval_knn(xTr.T, yTr.flatten(), lmnn.components_[0:3].T, xTe.T, yTe.flatten(), k)
    print('--> 1-NN Error for LMNN is : {}%'.format(100 * te_err))

    # ################################ 3-D Plot ###################################
    print("\n--> Plotting figures")
    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("PCA")
    pts_to_plt = L0[0:3] @ xTr

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax1.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("PCA-LDA")
    pts_to_plt = lda_xtr.T

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax2.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title("LMNN")
    pts_to_plt = lmnn.transform(xTr.T).T
    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax3.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title("GB-LMNN")

    plt.show()


if __name__ == '__main__':
    main()

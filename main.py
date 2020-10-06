# Iago Suarez implementation of Non-linear Metric Learning
import matplotlib.pyplot as plt
import numpy as np
from metric_learn import LMNN
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._ball_tree import BallTree
from sklearn.pipeline import Pipeline

from gb_lmnn import gb_lmnn
from knn import LSKnn2, eval_knn


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


def knn_classify_balltree(L, xTr, lTr, xTe, lTe, KK, tree=None, treesize=15, train=True, **kwargs):
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
    assert xTr.shape[0] == len(lTr)
    assert xTe.shape[0] == len(lTe)

    if len(L) != 0:
        # L is the initial linear projection, for example PCa or LDA
        xTr = xTr @ L.T
        xTe = xTe @ L.T

    if tree is None:
        print('Building tree ...')
        tree = BallTree(xTr, leaf_size=treesize, metric='euclidean')
    else:
        assert isinstance(tree, BallTree)

    MM = np.append(lTr, lTe).min()
    Kn = np.max(KK)
    assert np.alltrue(KK > 0)

    NTr, NTe = xTr.shape[0], xTe.shape[0]
    if xTr.shape[1] != xTe.shape[1]:
        raise ValueError('PROBLEM: Please make sure that training inputs and test inputs have the same dimensions!'
                         'xTr.shape: {}, xTe.shape: {}'.format(xTr.shape, xTe.shape))

    print("Evaluating...")
    Eval = np.zeros((2, KK))
    # Use the tree to compute the distance between the testing and training points
    # iTe: indices of the testing elements in the training set
    dists, iTe = tree.query(xTe, k=Kn, return_distance=True)
    assert iTe.ndim == 2 and iTe.shape == (NTe, 1)
    assert dists.ndim == 2 and dists.shape == (NTe, 1)

    # Labels of the testing elements in the training set
    lTe2 = LSKnn2(lTr[iTe], KK, MM)
    # Compute the error for each k
    Eval[1] = np.sum(lTe2 != np.repeat(lTe, KK, axis=0), axis=1) / NTe

    Details = {}
    if train:
        # Use the tree to compute the distance between the training points
        dists, iTr = tree.query(xTr, k=Kn + 1, return_distance=True)
        iTr = iTr[:, 1:]
        lTr2 = LSKnn2(lTr[iTr], KK, MM)
        Eval[0, :] = np.sum(lTr2 != np.repeat(lTr, KK, axis=0), axis=1) / NTr

        Details['lTr2'] = lTr2
        Details['iTr'] = iTr
    else:
        Eval[0] = []

    Details['lTe2'] = lTe2
    Details['iTe'] = iTe
    return Eval, Details


# #####################################################################################3

def main():
    print("--> ************************************************************************************")
    print("--> *************************** Metric Learning Demo ***********************************")
    print("--> ************************************************************************************")

    # Load variables
    print("--> Loading data")
    _, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('data/segment.mat').values()
    yTr, yTe, yVa = yTr.flatten().astype(int) - 1, yTe.flatten().astype(int) - 1, yVa.flatten().astype(int) - 1
    err, _ = knn_classify_balltree([], xTr.T, yTr, xTe.T, yTe, 1)

    print("--> Training pca...")
    L0 = pca(xTr, whiten=True)[0].T
    err, _ = knn_classify_balltree(L0[0:3], xTr.T, yTr, xTe.T, yTe, 1)
    print('1-NN Error after PCA in 3d is : {}%'.format(100 * err[1]))

    print("--> Training pca-lda...")
    pca_lda = Pipeline([('pca', PCA(n_components=5, whiten=True)),
                        ('lda', LinearDiscriminantAnalysis(n_components=3))])
    pca_lda.fit(xTr.T, yTr.flatten())
    pca_eigen_vals = np.diag(1 / np.sqrt(pca_lda[0].explained_variance_))
    pcalda_mat = pca_lda[1].scalings_[:, :3].T @ pca_eigen_vals @ pca_lda[0].components_

    print("--> Training lmnn...")
    lmnn = LMNN(init='pca', k=7, learn_rate=1e-6, verbose=False, n_components=3, max_iter=1000)
    lmnn.fit(xTr.T, yTr.flatten())

    print('Learning nonlinear metric with GB-LMNN ... ')
    L = pcalda_mat
    L = loadmat('data/lmnn2_L.mat')['L']  # Load the matlab matrix
    embed = gb_lmnn(xTr, yTr, 3, L, ntrees=200, verbose=True, xval=xVa, yval=yVa)
    # KNN classification error after metric learning using gbLMNN
    err, Details = knn_classify_balltree([], embed.transform(xTr.T), yTr, embed.transform(xTe.T), yTe, 1)
    print('GB-LMNN Training Error: {:.2f}%, Test (Error: {:.2f}%'.format(100 * err[0, 0], 100 * err[1, 0]))

    # ################################ k-NN evaluation ###################################
    print("\n--> Evaluation:")
    k = 1
    te_err = eval_knn(xTr.T, yTr.flatten(), np.eye(len(xTr)), xTe.T, yTe.flatten(), k)
    print('--> 1-NN Error for raw (high dimensional) input is : {:.2f}%'.format(100 * te_err))

    te_err = eval_knn(xTr.T, yTr.flatten(), L0[0:3].T, xTe.T, yTe.flatten(), k)
    print('--> 1-NN Error for PCA is : {:.2f}%'.format(100 * te_err))

    te_err = eval_knn(xTr.T, yTr.flatten(), pcalda_mat.T, xTe.T, yTe.flatten(), 1)
    print('--> 1-NN Error for PCA-LDA input is : {:.2f}%'.format(100 * te_err))

    te_err = eval_knn(xTr.T, yTr.flatten(), lmnn.components_[0:3].T, xTe.T, yTe.flatten(), k)
    print('--> 1-NN Error for LMNN is : {:.2f}%'.format(100 * te_err))

    te_err = eval_knn(embed.transform(xTr.T), yTr.flatten(), np.eye(3), embed.transform(xTe.T), yTe.flatten(), 1)
    print('--> 1-NN Error for GB-LMNN input is : {:.2f}%'.format(100 * te_err))

    # ################################ 3-D Plot ###################################
    print("\n--> Plotting figures")

    fig = plt.figure(figsize=plt.figaspect(1))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("PCA Training Error: {:.2f}, Testing Error: {:.2f}".format(100 * err[0, 0], 100 * err[1, 0]))
    pts_to_plt = L0[0:3] @ xTr

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax1.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("PCA-LDA")
    pts_to_plt = pcalda_mat @ xTr

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
    pts_to_plt = embed.transform(xTr.T).T
    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax4.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

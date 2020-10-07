"""
Pythonic implementation of the paper:
    Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
    Non-linear metric learning. In NIPS (pp. 2573-2581).
"""
__author__ = "Iago Suarez"
__email__ = "iago.suarez.canosa@alumnos.upm.es"

import matplotlib.pyplot as plt
import numpy as np
from metric_learn import LMNN
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from gb_lmnn import gb_lmnn
from knn import knn_error_score


def pca(X, whiten=False):
    """
    finds principal components of X.
    :param X:  dxn matrix (each column is a dx1 input vector)
    :param whiten: Whether to divide by the eigenvalues
    :return: (evects, evals)
        - evects  columns are principal components (leading from left->right)
        - evals   corresponding eigenvalues
    """
    X = X - X.mean(axis=1)[:, np.newaxis]
    cc = np.cov(X)
    cdd, cvv = np.linalg.eig(cc)
    ii = np.argsort(-cdd)
    evects = cvv[:, ii]
    evals = cdd[ii]
    if whiten:
        evects /= evals
    return evects, evals


# #####################################################################################

def main():
    print("************************************************************************************")
    print("*************************** Metric Learning Demo ***********************************")
    print("************************************************************************************")

    # Load variables
    print("Loading data")
    _, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('data/segment.mat').values()
    xTe, xTr, xVa = xTe.T, xTr.T, xVa.T
    yTr, yTe, yVa = yTr.flatten().astype(int) - 1, yTe.flatten().astype(int) - 1, yVa.flatten().astype(int) - 1

    print("Training pca...")
    L0 = pca(xTr.T, whiten=True)[0].T

    print("Training pca-lda...")
    pca_lda = Pipeline([('pca', PCA(n_components=5, whiten=True)),
                        ('lda', LinearDiscriminantAnalysis(n_components=3))])
    pca_lda.fit(xTr, yTr)
    pca_eigen_vals = np.diag(1 / np.sqrt(pca_lda[0].explained_variance_))
    pcalda_mat = pca_lda[1].scalings_[:, :3].T @ pca_eigen_vals @ pca_lda[0].components_

    print("Training lmnn...")
    lmnn = LMNN(init='pca', k=7, learn_rate=1e-6, verbose=False, n_components=3, max_iter=1000)
    lmnn.fit(xTr, yTr)

    print('Learning nonlinear metric with GB-LMNN ... ')
    # L = pcalda_mat
    L = loadmat('data/lmnn2_L.mat')['L']  # Load the matlab matrix
    embed = gb_lmnn(xTr, yTr, 3, L, n_trees=200, verbose=True, xval=xVa, yval=yVa)

    # ################################ k-NN evaluation ###################################
    print("\nEvaluation:")
    k = 1
    raw_tr_err, raw_te_err = knn_error_score(L0[0:3], xTr, yTr, xTe, yTe, k)
    print('1-NN Error for raw (high dimensional) input is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * raw_tr_err, 100 * raw_te_err))

    pca_tr_err, pca_te_err = knn_error_score(L0[0:3], xTr, yTr, xTe, yTe, k)
    print('1-NN Error for PCA in 3d is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * pca_tr_err, 100 * pca_te_err))

    lda_tr_err, lda_te_err = knn_error_score(pcalda_mat, xTr, yTr, xTe, yTe, k)
    print('1-NN Error for PCA-LDA input is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * lda_tr_err, 100 * lda_te_err))

    lmnn_tr_err, lmnn_te_err = knn_error_score(lmnn.components_[0:3], xTr, yTr, xTe, yTe, k)
    print('1-NN Error for LMNN is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * lmnn_tr_err, 100 * lmnn_te_err))

    gb_tr_err, gb_te_err = knn_error_score([], embed.transform(xTr), yTr, embed.transform(xTe), yTe, 1)
    print('1-NN Error for GB-LMNN input is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * gb_tr_err, 100 * gb_te_err))

    # ################################ 3-D Plot ###################################
    print("\nPlotting figures")

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("PCA Train Error: {:.2f}, Test Error: {:.2f}".format(100 * pca_tr_err, 100 * pca_te_err))
    pts_to_plt = xTr @ L0[0:3].T

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax1.scatter(pts_to_plt[mask, 0], pts_to_plt[mask, 1], pts_to_plt[mask, 2], label=l)
    plt.legend()

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("PCA-LDA Train Error: {:.2f}, Test Error: {:.2f}".format(100 * lda_tr_err, 100 * lda_te_err))
    pts_to_plt = xTr @ pcalda_mat.T

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax2.scatter(pts_to_plt[mask, 0], pts_to_plt[mask, 1], pts_to_plt[mask, 2], label=l)
    plt.legend()

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title("LMNN Train Error: {:.2f}, Test Error: {:.2f}".format(100 * lmnn_tr_err, 100 * lmnn_te_err))
    pts_to_plt = lmnn.transform(xTr)
    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax3.scatter(pts_to_plt[mask, 0], pts_to_plt[mask, 1], pts_to_plt[mask, 2], label=l)
    plt.legend()

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title("GB-LMNN Train Error: {:.2f}, Test Error: {:.2f}".format(100 * gb_tr_err, 100 * gb_te_err))
    pts_to_plt = embed.transform(xTr)
    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax4.scatter(pts_to_plt[mask, 0], pts_to_plt[mask, 1], pts_to_plt[mask, 2], label=l)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

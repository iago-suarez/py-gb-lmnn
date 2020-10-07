# Iago Suarez implementation of Non-linear Metric Learning
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


# #####################################################################################3

def main():
    print("--> ************************************************************************************")
    print("--> *************************** Metric Learning Demo ***********************************")
    print("--> ************************************************************************************")

    # Load variables
    print("--> Loading data")
    _, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('data/segment.mat').values()
    # TODO xTe, xTr, xVa = xTe.T, xTr.T, xVa.T
    yTr, yTe, yVa = yTr.flatten().astype(int) - 1, yTe.flatten().astype(int) - 1, yVa.flatten().astype(int) - 1

    print("--> Training pca...")
    L0 = pca(xTr, whiten=True)[0].T

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
    embed = gb_lmnn(xTr.T, yTr, 3, L, ntrees=200, verbose=True, xval=xVa.T, yval=yVa)

    # ################################ k-NN evaluation ###################################
    print("\n--> Evaluation:")
    k = 1
    raw_tr_err, raw_te_err = knn_error_score(L0[0:3], xTr.T, yTr, xTe.T, yTe, k)
    print('--> 1-NN Error for raw (high dimensional) input is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * raw_tr_err, 100 * raw_te_err))

    pca_tr_err, pca_te_err = knn_error_score(L0[0:3], xTr.T, yTr, xTe.T, yTe, k)
    print('--> 1-NN Error for PCA in 3d is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * pca_tr_err, 100 * pca_te_err))

    lda_tr_err, lda_te_err = knn_error_score(pcalda_mat, xTr.T, yTr, xTe.T, yTe, k)
    print('--> 1-NN Error for PCA-LDA input is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * lda_tr_err, 100 * lda_te_err))

    lmnn_tr_err, lmnn_te_err = knn_error_score(lmnn.components_[0:3], xTr.T, yTr, xTe.T, yTe, k)
    print('--> 1-NN Error for LMNN is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * lmnn_tr_err, 100 * lmnn_te_err))

    gb_tr_err, gb_te_err = knn_error_score([], embed.transform(xTr.T), yTr, embed.transform(xTe.T), yTe, 1)
    print('--> 1-NN Error for GB-LMNN input is, Training: {:.2f}%, Testing {:.2f}%'
          .format(100 * gb_tr_err, 100 * gb_te_err))

    # ################################ 3-D Plot ###################################
    print("\n--> Plotting figures")

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("PCA Train Error: {:.2f}, Test Error: {:.2f}".format(100 * pca_tr_err, 100 * pca_te_err))
    pts_to_plt = L0[0:3] @ xTr

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax1.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("PCA-LDA Train Error: {:.2f}, Test Error: {:.2f}".format(100 * lda_tr_err, 100 * lda_te_err))
    pts_to_plt = pcalda_mat @ xTr

    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax2.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title("LMNN Train Error: {:.2f}, Test Error: {:.2f}".format(100 * lmnn_tr_err, 100 * lmnn_te_err))
    pts_to_plt = lmnn.transform(xTr.T).T
    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax3.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_title("GB-LMNN Train Error: {:.2f}, Test Error: {:.2f}".format(100 * gb_tr_err, 100 * gb_te_err))
    pts_to_plt = embed.transform(xTr.T).T
    for l in np.unique(yTr):
        mask = np.squeeze(yTr == l)
        ax4.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

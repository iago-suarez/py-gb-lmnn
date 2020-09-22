# Iago Suarez implementation of Non-linear Metric Learning
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat


def buildmtreemex(x, mi):
    """
    Builds a tree.
    :param x:  input vectors (columns are vectors)
    :param mi: maximum number of points in leaf
    :return: (xoutput, index, treeinfo)
        - xoutput : input vectors reshuffled (sorted according to tree assignment)
        - index :	index of xouptput (ie xoutput=xinput(x) )
        - treeinfo : all the necessary information for the tree
    """
    tree = None
    # TODO
    return tree


def usemtreemexomp(xtest, xtrain, tree, k):
    """

    :param xtest:
    :param xtrain:
    :param tree:
    :param k:
    :return:
    """
    iknn, dists = None, None

    return iknn, dists


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
    Ni = Ni - MM - 1
    classes = np.unique(Ni)[:, np.newaxis]
    T = np.zeros((len(classes), N, KK))
    for i, c in enumerate(classes):
        for k in KK:
            T[i, :, k] = np.sum(Ni[0:k, :] == c, axis=0)

    yy = np.zeros((np.max(KK), N))
    for k in KK:
        # Select the maximum T among all the classes
        temp, yy[k, 0:N] = np.max(T[:, :, k] + T[:, :, 1] * 0.01, axis=1)
        # yy will contain the labels
        yy[k, 0:N] = classes[yy[k, :]]
    yy = yy[KK, :]
    yy = yy + MM - 1
    return yy


def pca(X):
    """
    finds principal components of
    :param X:  dxn matrix (each column is a dx1 input vector)
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
    return evects, evals


def knnclassifytreeomp(L, xTr, lTr, xTe, lTe, KK, **kwargs):
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
    assert xTr.shape[1] == len(lTr)
    assert xTe.shape[1] == len(lTe)

    train = 1 if 'train' not in kwargs else kwargs['train']
    if 'treesize' not in kwargs:
        kwargs['treesize'] = 15
    if len(L) == 0:
        dim = xTr.shape[0]
    else:
        # L is the initial linear projection, for example PCa or LDA
        dim = L.shape[1]
        xTr = L @ xTr
        xTe = L @ xTe

    if 'tree' in kwargs:
        # print('Building tree ...')
        kwargs['tree'].tree = buildmtreemex(xTr, kwargs['treesize'])
        # print('done\n')
    # TODO
    MM = np.append(lTr, lTe).min()
    Kn = np.max(KK)
    assert np.alltrue(KK > 0)

    NTr = xTr.shape[1]
    NTe = xTe.shape[1]

    # if(size(xTr,1)~=size(xTe,1))
    #  fprintf('PROBLEM: Please make sure that training inputs and test inputs have the same dimensions!\n');
    #  fprintf('size(xTr)');
    #  size(xTr)
    #  fprintf('size(xTe)');
    #  size(xTe)
    #  Eval=[];
    #  Details=[];
    #  return;
    # end;

    # if(max(max(pars.tree.jumpindex))~=NTr)
    #  fprintf('PROBLEM: Tree does not seem to belong to training data!\n');
    #  fprintf('Max index of tree: %i\n',max(max(pars.tree.jumpindex)));
    #  fprintf('Length of training data: %i\n',NTr);
    #  Eval=[];
    #  Details=[];
    #  return;
    # end;

    # TODO
    # if len(KK) == 1:
    #     outputK = KK
    #     KK = range(0, KK)
    # else:
    #     outputK = range(0, len(KK))
    outputK = 1

    Eval = np.zeros(2, KK)
    # Use the tree to compute the distance between the testing and training points
    # iTe: indices of the testing elements in the training set
    iTe, dists = usemtreemexomp(xTe, xTr, kwargs['tree'], Kn)
    assert iTe.ndim == 2 and iTe.shape == (1, NTe)
    assert dists.ndim == 2 and dists.shape == (1, NTe)

    # Labels of the testing elements in the training set
    lTe2 = LSKnn2(lTr[iTe].reshape((np.max(KK), NTe)), KK, MM)
    Eval[1] = np.sum(np.transpose(lTe != np.tile(lTe, KK)), axis=0) / NTe

    Details = {}
    if train:
        # Use the tree to compute the distance between the training points
        iTr, dists = usemtreemexomp(xTr, xTr, kwargs['tree'], Kn + 1)
        iTr = iTr[1:]
        lTr2 = LSKnn2(lTr(iTr), KK, MM)
        Eval[0, :] = np.sum(np.transpose(lTr2 != np.tile(lTr, KK, 1)), axis=1) / NTr

        Details['lTr2'] = lTr2
        Details['iTr'] = iTr
    else:
        # TODO
        Eval[0] = []

    Details['lTe2'] = lTe2
    Details['iTe'] = iTe
    Eval = Eval[:, outputK]
    return Eval, Details, kwargs['tree']


# #####################################################################################3

# Load variables
_, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('data/segment.mat').values()

# err, _, _ = knnclassifytreeomp([], xTr, yTr, xTe, yTe, 1)
#
# print('1-NN Error for raw (high dimensional) input is : {}%'.format(100 * err[1]))

print('\n')
L0 = pca(xTr)[0].T

# err, _, _ = knnclassifytreeomp(L0[0:3], xTr, yTr, xTe, yTe, 1)
# print('\n')
# print('1-NN Error after PCA in 3d is : {}%'.format(100 * err[1]))

# plt.subplot(3, 2, 1)

ax = Axes3D(plt.figure(1, figsize=(8, 6)), elev=-150, azim=110)
pts_to_plt = L0[0:3] @ xTr
for l in np.unique(yTr):
    mask = np.squeeze(yTr == l)
    ax.scatter(pts_to_plt[0, mask], pts_to_plt[1, mask], pts_to_plt[2, mask], label=l)
plt.legend()
plt.show()

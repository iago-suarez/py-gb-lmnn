# Iago Suarez implementation of Non-linear Metric Learning
import numpy as np

from mtrees import buildmtreemex, usemtreemex, getlayer, buildtree, evaltree, evalensemble, DataObject


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
    Eval, Details = None, None

    # TODO

    return Eval, Details


class Ensemble:

    @property
    def n_wls(self):
        return len(self.weak_learners)

    def __init__(self, weak_learners=None, learning_rates=None, L=None):
        if weak_learners is None:
            weak_learners = []
        if learning_rates is None:
            learning_rates = []
        self.weak_learners = weak_learners
        self.learning_rates = learning_rates
        self.L = L


def findtargetneighbors(X, labels, K, n_classes):
    targets_ind = None
    D, N = X.shape
    targets_ind = np.zeros((N, K))
    for i in range(n_classes):
        jj = np.where(labels == i)
        # Samples of the class i
        Xu = X[:, jj]
        T = buildmtreemex(Xu, 50)
        # Array of shape (4, len(Xu))
        targets = usemtreemex(Xu, Xu, T, K + 1)
        targets_ind[jj] = jj[targets[2:]].T

    return targets_ind


def findimpostors(pred, labels, n_classes, no_potential_impo):
    N = pred.shape[-1]
    active = np.zeros(no_potential_impo, N)
    for i in range(n_classes):
        mask = np.where(labels == i)
        pi = pred[:, mask]
        jj = np.where(~mask)
        pj = pred[:, jj]
        Tj = buildmtreemex(pj, 50)
        active[:, mask] = jj[usemtreemex(pi, pj, Tj, no_potential_impo)]

    return active


def lmnnobj(pred, targets_ind, active):
    n_dims, n_samples = pred.shape
    hinge, grad = np.zeros(n_samples), np.zeros((n_dims, n_samples))
    # TODO
    return hinge, grad


def gb_lmnn(X, Y, K, L, tol=1e-3, verbose=True, depth=4, ntrees=200, lr=1e-3, no_potential_impo=50,
            xval=np.array([]), yval=np.array([]), **kwargs):
    """
    Nonlinear metric learning using gradient boosting regression trees.
    :param X: (dxn) is the input training data, 'labels' (1xn) the corresponing labels
    :param Y: (kxd) is an initial linear transformation which can be learned using LMNN
    :param K: Number of nearest neighbours
    :param L: (kxd) is an initial linear transformation which can be learned using LMNN.
    corresponds to a metric M=L'*L
    :param tol: Tolerance for convergence
    :param verbose:
    :param depth: Tree depth
    :param kwargs:
    :param ntrees: number of boosted trees
    :param lr: learning rate for gradient boosting
    :param no_potential_impo:
    :param xval:
    :param yval:
    :return:
    """
    un, labels = np.unique(Y), Y
    embedding = None
    n_classes = len(un)

    pred = L @ X
    if len(xval) != 0:
        predVAL = L @ xval
        computevalerr = lambda pred, predVAL: knncl([], pred, Y, predVAL, yval, 1, train=False)
    else:
        predVAL = np.array([])
        computevalerr = lambda pred, predVAL: - 1.0

    # Initialize some variables
    D, N = X.shape
    assert (len(labels) == N)

    # find K target neighbors
    targets_ind = findtargetneighbors(X, labels, K, n_classes)

    # sort the training input feature-wise (column-wise)
    N = X.shape[1]
    #  sorts each column of x.T in ascending order.
    Xs, Xi = np.sort(X.T, axis=0), np.argsort(X.T, axis=0)

    # initialize ensemble (cell array of trees)
    ensemble = Ensemble()

    # initialize the lowest validation error
    lowestval = np.inf
    embedding = lambda: X

    # initialize roll-back in case stepsize is too large
    OC = np.inf
    Opred = pred
    OpredVAL = predVAL

    iter = 0
    # Perform main learning iterations
    while ensemble.n_wls < ntrees:
        # Select potential imposters
        if iter % 10 == 0:
            active = findimpostors(pred, labels, n_classes, no_potential_impo)
            OC = np.inf  # allow objective to go up

        hinge, grad = lmnnobj(pred, targets_ind.T.astype(np.int16), active.astype(np.int16))
        C = np.sum(hinge)
        print('--> It {} Loss value ({}) ...'.format(ensemble.n_wls, C))
        if C > OC:  # roll back in case things go wrong
            C = OC
            pred = Opred
            predVAL = OpredVAL
            # remove from ensemble
            ensemble.weak_learners.pop(), ensemble.learning_rates.pop()
            print('Learing rate too large ({}) ...'.format(lr))
            lr /= 2.0
        else:
            # Otherwise increase learning rate a little
            lr *= 1.01

        # Perform gradient boosting: construct trees to minimize loss
        tree, p = buildtree(X.T, Xs, Xi, -grad.T, depth, kwargs)

        # update predictions and ensemble
        Opred = pred
        OC = C
        OpredVAL = predVAL

        # Update predictions
        pred = pred + lr * p.T
        iter = ensemble.n_wls + 1

        # Add the tree and thew learning rate to the ensemble
        ensemble.weak_learners.append(tree)
        ensemble.learning_rates.append(lr)

        # update embedding of validation data
        if len(xval) > 0:
            predVAL = predVAL + lr * evaltree(xval.T, tree).T

        # Print out progress
        no_slack = np.sum(hinge > 0)
        if iter % 5 == 0 and verbose:
            print("Iteration {}: loss is {}, violating inputs: {}, learning rate: {}".format(
                iter, C / N, no_slack, lr))

        if iter % 10 == 0 or iter == (ntrees - 1):
            ensemble.L = L
            # TODO Review this code, looks strange
            newemb = evalensemble(X.T, ensemble, X.T * ensemble.L.T).T
            valerr = computevalerr(pred, predVAL)
            if valerr <= lowestval:
                lowestval = valerr
                embedding = newemb
                if verbose and lowestval >= 0.0:
                    print('Best validation error: {.2f}%'.format(lowestval * 100.0))
    return embedding

# Iago Suarez implementation of Non-linear Metric Learning
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from mtrees import usemtreemex, getlayer, buildtree, evaltree, evalensemble, DataObject, MTree


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
    D, N = X.shape
    targets_ind = np.zeros((N, K), dtype=int)
    for i in range(n_classes):
        jj, = np.where(labels == i)
        # Samples of the class i
        Xu = X[:, jj]
        T = MTree.build(Xu, 50)
        # Array of shape (4, len(Xu))
        targets, _ = usemtreemex(Xu, Xu, T, K + 1)
        targets_ind[jj] = jj[targets[1:]].T

    return targets_ind


def findimpostors(pred, labels, n_classes, no_potential_impo):
    print("  --> Finding impostors...")
    N = pred.shape[-1]
    active = np.zeros((no_potential_impo, N), dtype=int)
    for i in tqdm(range(n_classes)):
        ii, = np.where(labels == i)
        pi = pred[:, ii]
        jj, = np.where(labels != i)
        pj = pred[:, jj]
        # Use a tree Tj to search in hard negatives of class i
        Tj = MTree.build(pj, 50)
        active[:, ii] = jj[usemtreemex(pi, pj, Tj, no_potential_impo)[0]]

    return active


# def computeloss(X, & T[i * kt], & I[i * ki], d, kt, ki, i, grad):
def computeloss(X, T, I, d, kt, ki, i, grad):
    dt = np.zeros(kt)
    lossT = 0.0
    lossI = 0.0
    # compute distances to target neighbors
    for k in range(kt):
        dt[k] = np.sum((X[:, i] - X[:, T[k]]) ** 2) + 1
        # print("----------> dt[{}]: {}".format(k, dt[k]))
        lossT += dt[k] - 1.0
        # update gradient
        grad[:, i] += X[:, i] - X[:, T[k]]
        grad[:, T[k]] -= X[:, i] - X[:, T[k]]

    ma = dt.max()
    # print("-----------> ma: {}".format(ma))
    #  compute distances to impostors
    for k in range(ki):
        dis = min(ma, np.sum((X[:, i] - X[:, I[k]]) ** 2))
        # print("******> k: {}, dis: {}".format(k, dis))
        for t in range(kt):
            if dt[t] > dis:
                lossI += dt[t] - dis

                # Update gradient
                grad[:, i] -= X[:, T[t]] - X[:, I[k]]
                grad[:, T[t]] -= X[:, i] - X[:, T[t]]
                grad[:, I[k]] += X[:, i] - X[:, I[k]]
                # grad(:,i) = grad(:,i)-mu*(pred(:,targets_i)-pred(:,impos_i(k)));
                # grad(:,targets_i) = grad(:,targets_i)-mu*(pred(:,i)-pred(:,targets_i));
                # grad(:,impos_i(k)) = grad(:,impos_i(k))+mu*(pred(:,i)-pred(:,impos_i(k)));
    # print("----> lossT: {}, lossI: {}".format(lossT, lossI))
    return 0.5 * (lossT + lossI)


def lmnnobj(pred, targets_ind, active_ind):  # X, T, I
    """
    Computes the hinge loss and its gradient for the formula (8) of Non-linear Metric Learning:
    .. math::
        \mathcal{L}(\phi)=\sum_{i, j: j \sim i}\left\|\phi\left(\mathbf{x}_{i}\right)-\phi\left(\mathbf{x}_{j}\right)
        \right\|_{2}^{2}+\mu \sum_{k: y_{i} \neq y_{k}}\left[1+\left\|\phi\left(\mathbf{x}_{i}\right)-\phi\left(
        \mathbf{x}_{j}\right)\right\|_{2}^{2}-\left\|\phi\left(\mathbf{x}_{i}\right)-\phi\left(\mathbf{x}_{k}\right)
        \right\|_{2}^{2}\right]_{+}
    :param pred: Array of floats with shape (n_final_dims, n_samples).
    The actual points X projected on the target low-dimensional space.
    :param targets_ind: Array of integers with shape (n_final_dims, n_samples).
    Indices of target neighbors, the ones that we want to keep close.
    :param active_ind: Array of integers with shape (N_IMPOSTORS, n_samples). Impostor indices.
    :return: The evaluation of the loss and its gradient
    """
    n_dims, n_samples = pred.shape
    hinge, grad = np.zeros(n_samples), np.zeros((n_dims, n_samples))
    kt = targets_ind.shape[0]  # number of target neighbors
    ki = active_ind.shape[0]  # number of impostors
    for i in range(n_samples):
        # print("--> i: " + str(i))
        hinge[i] = computeloss(pred, targets_ind[:, i], active_ind[:, i], n_dims, kt, ki, i, grad)
        # print("--> hinge: {}, grad: {}".format(hinge[i], grad[:, i]))

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
    # Check that the labels have the correct format [0, 1, ..., L]
    assert np.alltrue(un == np.arange(len(un)))
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

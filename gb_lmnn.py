# Iago Suarez implementation of Non-linear Metric Learning
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors._ball_tree import BallTree
from sklearn.tree import DecisionTreeRegressor

from knn import eval_knn


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

    def eval(self, X):
        return self.transform(X)

    def transform(self, X):
        assert len(self.weak_learners) > 0, "Error: The model hasn't been trained"

        # extract from ensemble
        label_length = self.weak_learners[0].n_outputs_

        # initialize predictions
        n = X.shape[0]
        if self.L is None:
            p = np.zeros((n, label_length))
        else:
            p = X @ self.L.T

        # compute predictions from trees
        for i in range(self.n_wls):
            p += self.learning_rates[i] * self.weak_learners[i].predict(X)
        return p


def find_target_neighbors(X, labels, K, n_classes):
    X = X.T
    N, D = X.shape
    targets_ind = np.zeros((N, K), dtype=int)
    for i in range(n_classes):
        jj, = np.where(labels == i)
        # Samples of the class i
        Xu = X[jj]
        kdt = BallTree(Xu, leaf_size=50, metric='euclidean')
        targets = kdt.query(Xu, k=K + 1, return_distance=False)
        targets_ind[jj] = jj[targets[:, 1:]]

    return targets_ind


def find_impostors(pred, labels, n_classes, no_potential_impo):
    pred = pred.T
    N = len(pred)
    active = np.zeros((N, no_potential_impo), dtype=int)
    for i in range(n_classes):
        ii, = np.where(labels == i)
        pi = pred[ii]
        jj, = np.where(labels != i)
        pj = pred[jj]
        # Find the nearest neighbors using a BallTree
        kdt = BallTree(pj, leaf_size=50, metric='euclidean')
        hardest_examples = kdt.query(pi, k=no_potential_impo, return_distance=False)
        active[ii] = jj[hardest_examples]

    return active.T


def compute_loss(X, T, I, kt, i, grad):
    # TODO This uses a fixed margin of 1, this should be parametrized depending of the problem!
    # compute distances to target neighbors
    dt = cdist(X[i, np.newaxis], X[T], 'sqeuclidean').flatten()
    lossT = np.sum(dt)
    # compute the influence of the target neighbors in the gradient
    grad[i] += np.sum(X[np.newaxis, i] - X[T], axis=0)
    grad[T] -= X[i] - X[T]

    dists = cdist(X[i, np.newaxis], X[I], 'sqeuclidean').flatten()
    # Hinge loss
    lossI = np.maximum(0, 1 + dt[:, np.newaxis] - dists).sum()
    #  compute distances to impostors
    for k, dis in enumerate(dists):  # For each impostor
        for j in range(kt):  # For each target neighbor
            # For each target neighbor
            if dt[j] > dis - 1:
                # TODO Understand: Update gradient
                grad[i] -= X[T[j]] - X[I[k]]
                grad[T[j]] -= X[i] - X[T[j]]
                grad[I[k]] += X[i] - X[I[k]]

    return 0.5 * (lossT + lossI)


def lmnn_obj_loss(pred, targets_ind, active_ind):  # X, T, I
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
    assert pred.ndim == 2 and targets_ind.ndim == 2 and active_ind.ndim == 2
    assert pred.shape[1] == targets_ind.shape[1] == active_ind.shape[1]
    pred = pred.T
    n_samples, n_dims = pred.shape
    hinge, grad = np.zeros(n_samples), np.zeros(pred.shape)
    kt = targets_ind.shape[0]  # number of target neighbors
    for i in range(n_samples):
        hinge[i] = compute_loss(pred, targets_ind[:, i], active_ind[:, i], kt, i, grad)

    return hinge, grad.T


def gb_lmnn(X, Y, K, L, tol=1e-3, verbose=True, depth=4, ntrees=200, lr=1e-3, no_potential_impo=50,
            xval=np.array([]), yval=np.array([]), **kwargs) -> Ensemble:
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
    n_classes = len(un)

    use_validation = xval is not None
    pred = L @ X
    pred_val = L @ xval if use_validation else None
    eye = np.eye(pred.shape[0])
    if use_validation:
        valerr = eval_knn(pred.T, Y, eye, pred_val.T, yval.flatten(), k=1)
        print("--> Initial validation error: {:.2f}%".format(100 * valerr))

    # Initialize some variables
    D, N = X.shape
    assert (len(labels) == N)

    # find K target neighbors
    targets_ind = find_target_neighbors(X, labels, K, n_classes)

    # sort the training input feature-wise (column-wise)
    N = X.shape[1]

    # initialize ensemble (cell array of trees)
    ensemble = Ensemble(L=L)

    # initialize the lowest validation error
    lowestval = np.inf
    best_ensemble = None if use_validation else ensemble

    # initialize roll-back in case stepsize is too large
    last_cost = np.inf
    last_pred = pred
    last_pred_val = pred_val

    iter = 0
    # Perform main learning iterations
    while ensemble.n_wls < ntrees:
        # Select potential imposters
        if iter % 10 == 0:
            active = find_impostors(pred, labels, n_classes, no_potential_impo)
            last_cost = np.inf  # allow objective to go up

        hinge, grad = lmnn_obj_loss(pred, targets_ind.T, active)
        cost = np.sum(hinge)
        if cost > last_cost:  # roll back in case things go wrong
            cost = last_cost
            pred = last_pred
            pred_val = last_pred_val
            # remove from ensemble
            ensemble.weak_learners.pop(), ensemble.learning_rates.pop()
            print('-->\t Learning rate too large ({}) ...'.format(lr))
            lr /= 2.0
        else:
            # Otherwise increase learning rate a little
            lr *= 1.01

        # Train the weak learner tree
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X.T, -grad.T)
        wl_pred = tree.predict(X.T)

        # Record previous values
        last_pred, last_cost, last_pred_val = pred, cost, pred_val

        # Update predictions
        pred = pred + lr * wl_pred.T
        iter = ensemble.n_wls + 1

        # Add the tree and thew learning rate to the ensemble
        ensemble.weak_learners.append(tree)
        ensemble.learning_rates.append(lr)

        # Print out progress
        no_slack = np.sum(hinge > 0)
        if iter % 10 == 0 and verbose:
            print("--> Iteration {}: loss is {}, violating inputs: {}, learning rate: {:.6f}".format(
                iter, cost / N, no_slack, lr))

        # update best_ensemble of validation data
        if use_validation:
            pred_val = pred_val + lr * tree.predict(xval.T).T

            if iter % 10 == 0 or iter == (ntrees - 1):
                eye = np.eye(pred.shape[0])
                valerr = eval_knn(pred.T, Y, eye, pred_val.T, yval.flatten(), k=1)
                if valerr <= lowestval:
                    lowestval = valerr
                    best_ensemble = deepcopy(ensemble)
                    if verbose and lowestval >= 0.0:
                        print('----> Best validation error: {:.2f}%'.format(lowestval * 100.0))
    return best_ensemble

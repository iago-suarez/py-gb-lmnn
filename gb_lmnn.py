"""
Pythonic implementation of the paper:
    Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
    Non-linear metric learning. In NIPS (pp. 2573-2581).
"""
__author__ = "Iago Suarez"
__email__ = "iago.suarez.canosa@alumnos.upm.es"

from copy import deepcopy
from time import time

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors._ball_tree import BallTree
from sklearn.tree import DecisionTreeRegressor

from knn import knn_error_score


class Ensemble:
    """Ensemble class that predicts based on the weighted sum of weak learners."""

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


def find_random_target_neighbors(X, labels, K, n_classes):
    N, D = X.shape
    targets_ind = np.zeros((N, K), dtype=int)
    for i in range(n_classes):
        jj, = np.where(labels == i)
        random_targets = np.random.choice(jj, (len(jj), K))
        # Check if the random selection has some pair i-i
        colliding_elements = random_targets == jj[:, np.newaxis]
        n_colliding_elements = colliding_elements.sum()
        while n_colliding_elements > 0:
            # If so, replace these values
            random_targets[colliding_elements] = np.random.choice(jj, n_colliding_elements)
            colliding_elements = random_targets == jj[:, np.newaxis]
            n_colliding_elements = colliding_elements.sum()
        targets_ind[jj] = random_targets

    return targets_ind


def find_impostors(pred, labels, n_classes, no_potential_impo):
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

    return active


def compute_loss(X, T, I, i, grad, margin=1.0):
    # compute distances to target neighbors
    targets_distance = cdist(X[i, np.newaxis], X[T], 'sqeuclidean').flatten()
    lossT = np.sum(targets_distance)
    # compute the influence of the target neighbors in the gradient
    grad[i] += np.sum(X[np.newaxis, i] - X[T], axis=0)
    grad[T] += X[T] - X[i]

    dists = cdist(X[i, np.newaxis], X[I], 'sqeuclidean').flatten()
    # Hinge loss
    lossI = np.maximum(0, 1 + targets_distance[:, np.newaxis] - dists).sum()
    #  compute distances to impostors
    for k, k_distance in enumerate(dists):  # For each impostor
        for j, j_distance in zip(T, targets_distance):  # For each target neighbor
            if j_distance > k_distance - margin:
                grad[i] += X[I[k]] - X[j]
                grad[j] += X[j] - X[i]
                grad[I[k]] += X[i] - X[I[k]]

    return 0.5 * (lossT + lossI)


def lmnn_obj_loss(pred, targets_ind, active_ind, margin=1.0):
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
    assert pred.shape[0] == targets_ind.shape[0] == active_ind.shape[0]

    n_samples, n_dims = pred.shape
    hinge, grad = np.zeros(n_samples), np.zeros(pred.shape)
    for i in range(n_samples):
        hinge[i] = compute_loss(pred, targets_ind[i], active_ind[i], i, grad, margin)

    return hinge, grad


def hinge_loss(X, target_ind, impostor_ind, mu=1.0, margin=1.0):
    n_samples, n_dims = X.shape
    hinge = np.zeros(n_samples)

    dists = squareform(pdist(X, 'sqeuclidean'))
    all_target_dist = np.take_along_axis(dists, target_ind, axis=1)
    all_impostors_dist = np.take_along_axis(dists, impostor_ind, axis=1)
    sum_target_dists = np.sum(all_target_dist, axis=1)

    for i in range(n_samples):
        lossI = np.maximum(0, margin + all_target_dist[i, :, np.newaxis] - all_impostors_dist[i]).sum()
        hinge[i] = sum_target_dists[i] + mu * lossI
    return hinge.mean()


def violating_hinge_loss(pred, target_ind, impostor_ind, margin=1.0):
    dists = squareform(pdist(pred, 'sqeuclidean'))
    all_target_dist = np.take_along_axis(dists, target_ind, axis=1)
    all_impostors_dist = np.take_along_axis(dists, impostor_ind, axis=1)

    violating = 0
    for i in range(len(pred)):
        violating += np.any(margin + all_target_dist[i, :, np.newaxis] - all_impostors_dist[i] > 0)

    return violating


def find_best_alpha(pred, wl_pred, target_ind, impostor_ind, loss_f=hinge_loss):
    local_diff_step = 1e-6
    alpha_interval = (-1, 2)
    alpha_interval_width = max(alpha_interval) - min(alpha_interval)

    while alpha_interval_width > 1e-8:
        alpha = (max(alpha_interval) + min(alpha_interval)) / 2.0
        aprox_gradient = (loss_f(pred + (alpha + local_diff_step) * wl_pred, target_ind, impostor_ind) -
                          loss_f(pred + (alpha - local_diff_step) * wl_pred, target_ind, impostor_ind)) / \
                         (2 * local_diff_step)
        if aprox_gradient.mean() > 0:
            # Move to the left
            alpha_interval = (alpha_interval[0], alpha)
        else:  # gradient < 0:
            # Move to the right
            alpha_interval = (alpha, alpha_interval[1])
        alpha_interval_width = max(alpha_interval) - min(alpha_interval)

    return alpha


def gb_lmnn(X, y, k, L, verbose=False, depth=4, n_trees=200, lr=1e-3, no_potential_impo=50, subsample_rate=1.0,
            xval=np.array([]), yval=np.array([])) -> Ensemble:
    """
    Nonlinear metric learning using gradient boosting regression trees.
    :param X: (NxD) is the input training data, 'labels' (1xn) the corresponding labels
    :param y: (N) is an initial linear transformation which can be learned using LMNN
    :param k: Number of nearest neighbours used to do the train step.
    :param L: (kxd) is an initial linear transformation which can be learned using LMNN.
    corresponds to a metric M=L'*L
    :param verbose: Displays the training evolution
    :param depth: Tree depth
    :param n_trees: number of boosted trees
    :param lr: learning rate for gradient boosting
    :param no_potential_impo: The number of potential impostors that will be used to pergorm the gradient computation.
    :param xval: The validation samples
    :param yval: The validation labels
    :return:
    """
    assert len(X) == len(y) and X.ndim == 2 and y.ndim == 1
    assert len(xval) == 0 or (xval.ndim == 2 and yval.ndim == 1 and len(xval) == len(yval))
    assert len(xval) == 0 or X.shape[1] == xval.shape[1]

    un, labels = np.unique(y), y
    assert np.alltrue(un == np.arange(len(un))), "Error: labels should have format [1, 2, ..., C]"
    n_classes = len(un)

    use_validation = xval is not None
    pred = X @ L.T
    pred_val = xval @ L.T if use_validation else None
    if use_validation:
        tr_err, val_err = knn_error_score([], X, y, xval, yval, k=1)
        print("Initial Training error: {:.2f}%, Val. error: {:.2f}%".format(100 * tr_err, 100 * val_err))

    # Initialize some variables
    N, D = X.shape

    # find K target neighbors
    targets_ind = find_target_neighbors(X, labels, k, n_classes)

    # initialize ensemble (cell array of trees)
    ensemble = Ensemble(L=L)

    # initialize the lowest validation error
    lowest_val_err = np.inf
    best_ensemble = deepcopy(ensemble)
    margin = 1.0

    # Perform main learning iterations
    while ensemble.n_wls < n_trees:
        start = time()
        # Select potential imposters
        impostor_ind = find_impostors(pred, labels, n_classes, no_potential_impo)
        hinge, grad = lmnn_obj_loss(pred, targets_ind, impostor_ind, margin)
        cost = np.sum(hinge)

        # Determine if we are going to use subsample
        if subsample_rate == 1.0:
            subsample_ind = slice(None)
        else:
            subsample_ind = np.random.randint(0, N, int(N * subsample_rate))

        # Train the weak learner tree
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X[subsample_ind], -grad[subsample_ind])
        wl_pred = tree.predict(X)

        alpha = lr  # * find_best_alpha(pred, wl_pred, targets_ind, impostor_ind)

        # Update predictions
        pred = pred + alpha * wl_pred

        # Add the tree and thew learning rate to the ensemble
        ensemble.weak_learners.append(tree)
        ensemble.learning_rates.append(alpha)

        # if iter % 10 == 0 and verbose:
        # Print out progress
        elapsed = time() - start
        iter = ensemble.n_wls + 1
        if verbose:
            print("Iteration {}: loss is {:.6f}, violating inputs: {}, alpha: {:.6f}, in {:.2f}s".format(
                iter, cost / N, violating_hinge_loss(pred, targets_ind, impostor_ind, margin), alpha, elapsed))

        # update best_ensemble of validation data
        if use_validation:
            pred_val = pred_val + alpha * tree.predict(xval)

            if iter % 5 == 0 or iter == (n_trees - 1):
                tr_err, val_err = knn_error_score([], pred, y, pred_val, yval, k=1)
                if verbose:
                    print("Iteration {}: Training error: {:.2f}%, Val. error: {:.2f}%".format(
                        iter, 100 * tr_err, 100 * val_err))

                if val_err <= lowest_val_err:
                    lowest_val_err = val_err
                    best_ensemble = deepcopy(ensemble)
                    if verbose:
                        print('--->\t\tBest validation error! :D')
    return best_ensemble

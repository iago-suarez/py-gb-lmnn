# Regression tree implementation based on the original one from:
# Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
# Non-linear metric learning. In NIPS (pp. 2573-2581).
import random
from collections import deque
from copy import deepcopy

import numpy as np


class MTree:
    def __init__(self, index, pivots_x, radius, jumpindex, kids, n_nodes, max_leaf_samples):
        """
        Regression tree
        :param index: (n_samples, ) New indices of points (random permutation of inputs)
        :param pivots_x: (19, n_nodes) locations of pivots
        :param radius: (n_nodes, ) radius of trees
        :param jumpindex: (2, n_nodes) array. jumpindex of trees (tree i goes from ji[0, i]:ji[1, i] )
        :param kids: (2, n_nodes) kids of trees (tree i has kids kids[0, i] and kids[1, i])
        :param n_nodes: Number of nodes in the tree.
        """
        self.pivots_x = pivots_x
        self.radius = radius
        self.jumpindex = jumpindex
        self.kids = kids
        self.index = index
        self.n_nodes = n_nodes
        self.max_leaf_samples = max_leaf_samples

    @classmethod
    def build(cls, X, max_leaf_samples=15):
        """
        Builds a new MTree. The policy to build the tree is to split the data until each leaf contains has less
         than max_leaf_samples elements samples. To split the data in two braches we pseudo-randomly select two pivot
         samples in the extremes of the branch and split the elements in closer to pivot1 and to pivot2.

        :param X: input vectors (rows are vectors)
        :param max_leaf_samples: maximum number of points in leaf
        :return: A new tree
        """

        MIN_RADIUS = 0.0001

        class TreeNode:
            def __init__(self, number=None, ij=None):
                """
                A node of the tree that contains a
                :param number: The identifier of the node
                :param ij: The indices over the array of samples that point to the start and end of the tree region.
                """
                self.number = number
                self.ij = ij

            def __str__(self):
                return "number: {}, ij [{}, {}]".format(self.number, self.ij[0], self.ij[1])

        # TODO avoid this transpose
        X = X.T.copy()

        N, DIM = X.shape
        tree = MTree(index=np.zeros(N, dtype=int),
                     pivots_x=np.zeros((DIM, N)),
                     radius=np.zeros(N),
                     jumpindex=np.zeros((2, N), dtype=int),
                     kids=np.zeros((2, N), dtype=int),
                     n_nodes=0,
                     max_leaf_samples=max_leaf_samples)

        # Initialize some variables
        index = np.arange(N)
        tree.n_nodes = 0
        s = deque()
        c = TreeNode(0, (0, N - 1))
        s.append(deepcopy(c))
        while len(s) > 0:
            random.seed(0)
            c = s.pop()  # pop first element from stack
            i1, i2 = c.ij[0], c.ij[1]  # set first and last index
            ni = i2 - i1 + 1  # compute length of interval
            # Select the data that fall in the current node
            node_indices = index[i1:(i2 + 1)]
            node_X = X[node_indices]
            # The pivot is the mean of the samples in the interval
            piv = np.mean(node_X, axis=0)  # get memory for pivot
            # Compute radius of ball. Finds the maximum L2 distance between vector piv and all rows in matrix x
            radius = np.sqrt(np.sum((node_X - piv) ** 2, axis=1).max())

            # Set node parameters
            tree.jumpindex[:, c.number] = [i1 + 1, i2 + 1]
            tree.radius[c.number] = radius
            tree.pivots_x[:, c.number] = piv

            if ni < max_leaf_samples or radius < MIN_RADIUS:
                # if tree has fewer than max_leaf_samples data points or a very small radius, make it a leaf
                tree.kids[:, c.number] = [-1, -1]  # indicate leaf node (through -1 kids)
            else:
                # compute statistics about pivot points
                tree.kids[:, c.number] = [tree.n_nodes + 2, tree.n_nodes + 3]
                # pick two points that are far away from each other
                r = np.random.randint(0, ni)
                # Index of point far away from r. Finds the maximum L2 distance between vector
                # node_X[r] and the cols in X.
                pivot1 = np.argmax(np.sum((node_X - node_X[r]) ** 2, axis=1))
                # Index of point fra away from pivot1
                pivot2 = np.argmax(np.sum((node_X - node_X[pivot1]) ** 2, axis=1))
                # compute dir=(x1-x2) and project each point onto this direction
                dir = node_X[pivot1] - node_X[pivot2]
                ips = node_X @ dir
                # decide if the projected point is closer to pivot 1 or pivot 2
                closer_to_pivot1 = ips > ips.mean()
                closer_to_pivot2 = ~closer_to_pivot1
                c1, c2 = np.sum(closer_to_pivot1), np.sum(closer_to_pivot2)
                # Fill the first elements of ind1 with the values of index where ips > np.mean(ips)
                index[i1:(i2 + 1)] = np.append(node_indices[closer_to_pivot1],
                                               node_indices[closer_to_pivot2])

                # Prevent potential infinite loop
                if c1 == 0 or c2 == 0:
                    raise Exception("A subtree with 0 elements was created. This should never happen!")

                # push subtree 1 onto the stack
                s.append(TreeNode(tree.n_nodes + 1, [i1, i1 + c1 - 1]))
                # push subtree 2 onto the stack
                s.append(TreeNode(tree.n_nodes + 2, [i1 + c1, i2]))
                tree.n_nodes += 2

        tree.index = index + 1
        tree.cut_off_zeros()
        ########################################################################
        return tree

    def fit(self, X, y):
        print("--> Training my tree")
        pass

    def predict(self, X):
        print("--> Using my tree")
        return 1.2

    def cut_off_zeros(self):
        self.pivots_x = self.pivots_x[:, :self.n_nodes]
        self.radius = self.radius[:self.n_nodes]
        self.jumpindex = self.jumpindex[:, :self.n_nodes]
        self.kids = self.kids[:, :self.n_nodes]


class DataObject:

    def __init__(self, numfeatures, y, n, depth, tree):
        self.numfeatures = numfeatures
        self.y = y
        self.n = n
        self.depth = depth
        self.tree = tree


def buildlayer_sqrimpurity_openmp(Xs, Xi, y, n, f, args):
    # This program chooses the best splits, finds the predictions, and computes the loss for the tree.
    raise NotImplementedError()


def preprocesslayer_sqrimpurity(data: DataObject, options):
    # confirm necessary data or options
    assert isinstance(data, DataObject)

    # compute counts for each node
    numnodes = 2 ** (data.depth - 1)
    # Allocating space for l_infty.
    m_infty = -99999 * np.ones((1, numnodes))
    l_infty = -99999 * np.ones((numnodes, data.y.shape[1]))
    for i in range(numnodes):
        m_infty[i] = np.sum(data.n == i)
        l_infty[i] = np.sum(data.y[data.n == i], axis=1)  # TODO is the axis right?

    l_infty = l_infty.T

    # get parents
    parents = getlayer(data.tree, data.depth)

    # include feature costs
    if 'featurecosts' in options:
        featurecosts = options['featurecosts']
    else:
        featurecosts = np.zeros(data.numfeatures, 1)

    return m_infty, l_infty, parents[:, 4:].T, featurecosts


def usemtreemex(xtest, xtrain, tree, k):
    iknn, dists = None, None
    # TODO
    return iknn, dists


def getlayer(tree, depth):
    # Tree implementation with arrays
    # TODO Ensure that this is valid for the 0...N-1 numpy indexing
    maxdepth = np.log2(tree.shape[0] + 1)
    rowindices = np.arange(2 ** (depth - 1), 2 ** depth - 1)
    layer = tree[rowindices, :]
    return layer, rowindices, maxdepth


def evaltree(X, tree):
    p = None
    # TODO
    return p


def buildtree(X, Xs, Xi, y, depth, kwargs):
    tree, p = None, None
    # Handling the multilabel case where gradient is vectorized.
    if X.shape[0] < y.shape[0]:
        if y.shape[0] % X.shape[0] != 0:
            raise Exception('Gradient elements count must divide with number of instances')
        else:
            y = y.reshape(len(X), -1)

    # verify agreement among X, Xs, Xi, and g
    # %Checks sizes of Xs, Xi, and y
    assert X.shape == Xs.shape == Xi.shape, 'Dimensions of X, Xs, and Xi do not match'
    assert len(X) == len(y), 'Dimensions of X and y do not match'
    n_samples = len(X)
    RowsX, numfeatures = X.shape
    RowsXs, col = Xs.shape

    # Cheks that the row dimensions of X
    if RowsX != RowsXs:
        raise Exception('Row dimentions in X do not match row dimensions in Xs, Xi, or y')

    #  initialize with each instance at the root node
    n = np.ones((n_samples, 1))

    # initialize tree and compute default label
    defaultlabel = kwargs['defaultlabel'](y) if 'defaultlabel' in kwargs else np.mean(y, axis=0)

    # initialize the root node with default prediction
    tree = [np.zeros(1, 3), defaultlabel]

    #  select function to build layer
    buildlayer = kwargs['buildlayer'] if 'buildlayer' in kwargs else buildlayer_sqrimpurity_openmp

    # select function to preprocess layer construction
    preprocesslayer = kwargs['preprocesslayer'] if 'preprocesslayer' in kwargs else preprocesslayer_sqrimpurity

    # build tree layer-wise
    for d in range(depth - 1):
        # initialize stuff
        splits, impurity, labels = [], [], []
        # get parent nodes
        parents = getlayer(tree, d)

        # prepare data for preprocessing
        data = DataObject(X.shape[1], y, n, d, tree)

        # compute preprocessed arguments
        args = preprocesslayer(data, kwargs)
        f = np.arange(0, numfeatures)
        # Splits says for each feature the best split points, impurity the impurity leve and
        # labels has shape (n_features, n_classes)
        splits, impurity, labels = buildlayer(Xs, Xi, y.T, n, f, args)

        # pick best splits for each node
        bestimpurity, bestfeatures = impurity.min(), np.argmin(impurity)
        # TODO Is this correct?
        # SUB2IND(SIZ,I,J) returns the linear index equivalent to the row and column subscripts in the arrays I and J for a matrix of size SIZ.
        # indices = sub2ind(size(impurity),bestfeatures,1:size(impurity,2));
        indices = np.unravel_index(np.arange(0, impurity.shape[1]), impurity.shape)
        bestsplits = splits[indices]

        # record splits for parent nodes
        pi = getlayer(tree, d)
        tree[pi, 0:3] = [bestfeatures, bestsplits, bestimpurity]

        # record labels for child nodes
        children = np.zeros((2, parents.shape[1]))
        bfpairs = np.tile(bestfeatures, 2 * (parents.shape[1] - 3))
        bfpairs = bfpairs.T
        # TODO Is this correct? sub2ind(size(labels),bfpairs,1:size(labels,2))';
        pairedindices = np.unravel_index(np.arange(0, labels.shape[1]), labels.shape)
        pairedindices = pairedindices.reshape((children.shape[1] - 3), children.shape[0]).T
        # pairedindices has shape (2, 3)
        children[:, 4:] = labels[pairedindices]  # size(tree) = 2, 6
        tree = np.vstack([tree, children])

        # Update nodes
        # Fv(n) = Feature Index
        # Vv(n) = Feature Value
        # Sv(n) = Split Value
        bestfeatures = bestfeatures.T
        Fv = bestfeatures[n]
        bestsplits = bestsplits.T
        Sv = bestsplits[n]
        # TODO  sub2ind(size(X),1:size(X,1),Fv');
        Iv = np.unravel_index(np.arange(0, X.shape[0]), X.shape)
        Vv = X[Iv]
        # TODO ?? Vv = Vv # to column vector
        n = 2 * n - (Vv < Sv)

    # update predictions
    leafnodes = getlayer(tree, depth)  # Stores the last Layer of tree into LL
    outputlabels = leafnodes[:, 4:]  # Stores the predictions of the tree into pp
    p = outputlabels[n]  # Stores the prediction for each instance into p

    return tree, p


def evalensemble(X, ensemble, p):
    p = None
    # TODO
    return p

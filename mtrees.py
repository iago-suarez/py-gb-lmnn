# Regression tree implementation based on the original one from:
# Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
# Non-linear metric learning. In NIPS (pp. 2573-2581).
import numpy as np


class TreeInfo:
    def __init__(self, index, piv, radius, jumpindex, kids):
        self.index = index
        self.piv = piv
        self.radius = radius
        self.jumpindex = jumpindex
        self.kids = kids


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


def buildmtreemex(x, mi):
    """

    :param x: input vectors (columns are vectors)
    :param mi: maximum number of points in leaf
    :return: A struct containing:
        - index: [1×198 double]
        - piv: [19×11 double]
        - radius: [154.2278 73.6423 98.4881 69.5657 83.9020 46.9396 40.5655 49.0868 72.0262 37.8169 50.5862]
        - jumpindex: [2×11 double]
        - kids: [2×11 double]
    """
    tree = TreeInfo(None, None, None, None, None)
    # TODO
    return tree


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

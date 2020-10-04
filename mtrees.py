# Regression tree implementation based on the original one from:
# Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
# Non-linear metric learning. In NIPS (pp. 2573-2581).
import random
from collections import deque

import numpy as np
from tqdm import tqdm


class MinHeapTree:

    @property
    def heapsize(self):
        return len(self.elements)

    @property
    def heapnodes(self):
        return np.array([e[0] for e in self.elements])

    @property
    def heapdata(self):
        return np.array([e[1] for e in self.elements])

    def __init__(self, d):
        self.heapmaxsize = d
        self.elements = []

    def heapupdate(self, key, data):
        self.elements.append((key, data))
        self.elements.sort(key=lambda e: e[0])
        self.elements = self.elements[:self.heapmaxsize]


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
        tree = MTree(index=np.arange(N, dtype=int),
                     pivots_x=np.zeros((DIM, N)),
                     radius=np.zeros(N),
                     jumpindex=np.zeros((2, N), dtype=int),
                     kids=np.zeros((2, N), dtype=int),
                     n_nodes=0,
                     max_leaf_samples=max_leaf_samples)

        # Initialize some variables
        tree.n_nodes = 0
        s = deque()  # This stack will contain the elements that we have to process
        s.append(TreeNode(0, (0, N - 1)))  # Add the root node of the tree
        while len(s) > 0:
            random.seed(0)
            c = s.pop()  # pop first element from stack
            i1, i2 = c.ij[0], c.ij[1]  # set first and last index
            ni = i2 - i1 + 1  # compute length of interval
            # Select the data that fall in the current node
            node_indices = tree.index[i1:(i2 + 1)]
            node_X = X[node_indices]
            # The pivot is the mean of the samples in the interval
            piv = np.mean(node_X, axis=0)  # get memory for pivot
            # Compute radius of ball. Finds the maximum L2 distance between vector piv and all rows in matrix x
            radius = np.sqrt(np.sum((node_X - piv) ** 2, axis=1).max())

            # Set node parameters
            tree.jumpindex[:, c.number] = [i1, i2]
            tree.radius[c.number] = radius
            tree.pivots_x[:, c.number] = piv

            if ni < max_leaf_samples or radius < MIN_RADIUS:
                # if tree has fewer than max_leaf_samples data points or a very small radius, make it a leaf
                tree.kids[:, c.number] = [-1, -1]  # indicate leaf node (through -1 kids)
            else:
                # compute statistics about pivot points
                tree.kids[:, c.number] = [tree.n_nodes + 1, tree.n_nodes + 2]
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
                tree.index[i1:(i2 + 1)] = np.append(node_indices[closer_to_pivot1],
                                                    node_indices[closer_to_pivot2])

                # Prevent potential infinite loop
                if c1 == 0 or c2 == 0:
                    raise Exception("A subtree with 0 elements was created. This should never happen!")

                # push subtree 1 onto the stack
                s.append(TreeNode(tree.n_nodes + 1, [i1, i1 + c1 - 1]))
                # push subtree 2 onto the stack
                s.append(TreeNode(tree.n_nodes + 2, [i1 + c1, i2]))
                tree.n_nodes += 2

        tree.n_nodes += 1
        tree.cut_off_zeros()
        ########################################################################
        return tree

    def _findknn(self, X, X_test, k):
        assert X.ndim == 2 and X_test.ndim == 1 and X.shape[1] == len(X_test)
        # The stack will contain pairs in format (node, distance)
        stack = deque()
        stack.append((0, 0))
        heap = MinHeapTree(k)

        while True:
            try:
                while True:
                    node, mindist = stack.pop()
                    fb = len(stack)
                    # If we dont have at least k neighbors, there is no more elements
                    # in the stack or mindist is bigger than the worse neighbor
                    if heap.heapsize != k or fb < 0 or mindist <= heap.elements[-1][0]:
                        # Break internal loop
                        break
            except IndexError:
                break

            # print(f"--> node: {node}, indices: [{self.jumpindex[0, node]}, {self.jumpindex[1, node]}]")
            kid1 = self.kids[0, node]
            if kid1 < 0:  # leaf
                # print("   --> Reached leaf node")
                dists = np.linalg.norm(X[self.jumpindex[0, node]:(1 + self.jumpindex[1, node])] - X_test, axis=1)
                if k == 1:
                    i = np.argmin(dists)
                    heap.heapupdate(dists[i], self.jumpindex[0, node] + i)
                else:
                    i_s = np.argsort(dists)[:k]
                    for i in i_s:
                        heap.heapupdate(dists[i], self.jumpindex[0, node] + i)

            else:
                kid2 = self.kids[1, node]
                d1 = np.linalg.norm(self.pivots_x[:, kid1] - X_test)
                d2 = np.linalg.norm(self.pivots_x[:, kid2] - X_test)
                # print("  --> kid1: {}, kid2: {}, d1: {:.3f}, d2: {:.3f}".format(kid1, kid2, d1, d2))
                if d1 < d2:
                    # if kid1 is the closer child
                    stack.append((kid2, max(d2 - self.radius[kid2], 0)))
                    stack.append((kid1, max(d1 - self.radius[kid1], 0)))
                else:
                    # if kid2 is the closer child
                    stack.append((kid1, max(d1 - self.radius[kid1], 0)))
                    stack.append((kid2, max(d2 - self.radius[kid2], 0)))

        return heap.heapdata, heap.heapnodes

    def findknn(self, X, X_test, k):
        X, X_test = X.T[self.index], X_test.T

        # do some basic checks:
        if X_test.shape[1] != X.shape[1]:
            raise ValueError('Training and Test data must have same dimensions!')

        assert self.pivots_x.shape[0] == X.shape[1] == X_test.shape[1]

        # Get the number of elements in the input argument.
        N, DIM = X_test.shape
        iknn = np.zeros((N, k), dtype=int)
        dists = np.zeros((N, k), dtype=float)
        for i in range(N):
            # print(f"Procesing sample i = {i}")
            iknn[i], dists[i] = self._findknn(X, X_test[i], k)
            # print(f" [found NN: {iknn[i]}, d: {dists[i]}]")

        # TODO Avoid transpose
        iknn = self.index.astype(int)[iknn.T]
        return iknn, dists.T

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
        # vector of shape (N,) that defines the index of the leaf node that predicts each example.
        self.n = n
        self.depth = depth
        self.tree = tree


class StaticNode:
    def __init__(self, m_infty_, label_length_, l_infty_):
        self.label_length = label_length_
        self.m_infty = m_infty_
        self.l_infty = l_infty_
        # Best split point found for this node
        self.split = 0.0
        # Best loss value found for this node
        self.loss = np.inf
        # m_s is a counter. The number of data points encountered so far that correspond to that node
        self.m_s = 0
        # # The previous feature value. It is used to select as split point.
        self.previous_xs = 0.0
        self.label = np.zeros(label_length_)
        # # l_s is the total residual encountered so far corresponding to that node.
        self.l_s = np.zeros(label_length_)

    @classmethod
    def create_from_children(cls, label_length_):
        return cls(0.0, label_length_, np.zeros(label_length_))


def feature_sqrimpurity_openmp_multi(Xs, Xi, Y, N, m_infty, l_infty, parents_labels, feature_cost):
    numnodes = m_infty.shape[0]
    assert N.max() < numnodes
    numinstances = Xs.shape[0]
    target_ndims = l_infty.shape[0]  # The number of dimensions in the output space

    first = True
    # instantiate parent and child layers of nodes
    parents, children = [], []
    for i in range(numnodes):
        parents.append(StaticNode(int(m_infty[i]), target_ndims, l_infty[:, i]))

        child1 = StaticNode.create_from_children(target_ndims)
        child1.label = parents_labels[:, i]
        children.append(child1)

        child2 = StaticNode.create_from_children(target_ndims)
        child2.label = parents_labels[:, i]
        children.append(child2)

    # iterate over examples
    for j in range(numinstances):
        # get current value
        v = Xs[j]  # feature value from training set value
        i = Xi[j]  # feature index that corresponds to the unsorted feature
        node_index = N[i]  # node index on the parent layer for the instance
        node = parents[node_index]  # node on the parent layer for the instance

        # If not first instance at node and greater than split point, consider new split at v
        if node.m_s > 0 and v > node.previous_xs:
            # compute split impurity
            l_s_sqrnorm = np.sum(node.l_s ** 2)
            l_infty_minus_l_s_sqrnorm = np.sum((node.l_infty - node.l_s) ** 2)

            loss_i = - l_s_sqrnorm / node.m_s \
                     - l_infty_minus_l_s_sqrnorm / (node.m_infty - node.m_s) \
                     + feature_cost
            # print(f"****> impurity: l_s_sqrnorm = {l_s_sqrnorm}, m_s = {node.m_s}, "
            #       f"l_infty_minus_l_s_sqrnorm = {l_infty_minus_l_s_sqrnorm}, "
            #       f"m_infty - m_s = {(node.m_infty - node.m_s)}, feature_cost = {feature_cost}")

            # compare with best and record if better
            if loss_i < node.loss:
                node.loss = loss_i
                # The split point is the one between the actual best and the previous
                node.split = 0.5 * (node.previous_xs + v)
                # Set the labels for the node children
                children[2 * node_index].label = node.l_s / node.m_s
                children[2 * node_index + 1].label = (node.l_infty - node.l_s) / (node.m_infty - node.m_s)

        # Update variable
        node.m_s += 1  # m_s is a counter. The number of data points encountered so far that correspond to that node
        node.l_s += Y[:, i]  # ls is the total residual encountered so far corresponding to that node.
        node.previous_xs = v  # The previous feature value.

    # Record output values for feature f
    return [p.split for p in parents], \
           [p.loss for p in parents], \
           np.array([c.label for c in children])


def buildlayer_sqrimpurity(Xs, Xi, Y, N, features, m_infty, l_infty, parents_labels, feature_cost, reshape_labels=True):
    """
    This program chooses the best splits, computes the loss, and finds the predictions for the tree.
    :param Xs: Sorted samples in original space
    :param Xi: Indices used to sort the elements in Xs
    :param Y: Labels or Residuals
    :param N: array of size Nx1, each element have the index of the node where that example falls.
    :param features: Range of features to be used (Feature Index)
    :param m_infty: Array with the number of samples predicted by the node.
    :param l_infty: Array with the sum of the predictions for all the samples predicted by the node.
    :param parents_labels:
    :param feature_cost:
    :return: splits, losses, labels
     Splits says for each feature the best split points, impurity the sqrimpurity level
     associated with that split-point and labels the prediction of the leaf node.
     label of child 1 contains: node->l_s / node->m_s
     label of child 2 contains: (node->l_infty - node->l_s) / (node->m_infty - node->m_s)
     Where:
       - node->l_s is the total residual encountered so far at that node.
       - node->m_infty is the number of samples predicted by the node.
       - node->l_infty is the sum of the predictions for all the samples predicted by the node.
       - node->m_s is the number of data points encountered so far at that node
    """
    assert Xs.ndim == 2 and Xi.ndim == 2 and Xs.shape == Xi.shape
    assert Y.ndim == 2 and Y.shape[1] == Xs.shape[0]
    assert N.ndim == 1 and len(N) == Xs.shape[0]
    assert m_infty.ndim == 1 and l_infty.ndim == 2
    numnodes = m_infty.shape[0]
    numfeatures = len(features)
    label_length = l_infty.shape[0]

    # Create outputs
    splits = np.zeros((numfeatures, numnodes))
    losses = np.zeros((numfeatures, numnodes))
    labels = np.zeros((numfeatures, 2 * numnodes, label_length))

    for f in features.flatten():
        # Calculate for this feature the split point with the best square impurity level
        splits[f], losses[f], labels[f] = feature_sqrimpurity_openmp_multi(
            Xs[:, f], Xi[:, f], Y, N, m_infty, l_infty, parents_labels, feature_cost[f])

    if reshape_labels:
        labels = labels.reshape(numfeatures, -1)
    return splits, losses, labels


def preprocesslayer_sqrimpurity(data: DataObject, featurecosts=None):
    # confirm necessary data or options
    assert isinstance(data, DataObject)

    # compute counts for each node
    numnodes = 2 ** (data.depth - 1)
    # Allocating space for l_infty.
    m_infty = -99999 * np.ones(numnodes)
    l_infty = -99999 * np.ones((numnodes, data.y.shape[1]))
    for i in range(numnodes):
        m_infty[i] = np.sum(data.n == i)
        l_infty[i] = np.sum(data.y[data.n == i], axis=0)

    l_infty = l_infty.T

    # get parents
    parents = getlayer(data.tree, data.depth)[0]

    # include feature costs
    if featurecosts is None:
        featurecosts = np.zeros(data.numfeatures)

    return m_infty, l_infty, parents[:, 3:].T, featurecosts


# TODO Delete
def usemtreemex(xtest, xtrain, tree, k):
    if xtest.shape[0] != xtrain.shape[0]:
        raise ValueError('Training and Test data must have same dimensions!')

    dim = tree.pivots_x.shape[0]
    return tree.findknn(xtrain[:dim, tree.index], xtest[:dim], k)


def getlayer(tree, depth):
    # Tree implementation with arrays
    # TODO Ensure that this is valid for the 0...N-1 numpy indexing
    maxdepth = np.log2(tree.shape[0] + 1)
    rowindices = np.arange(2 ** (depth - 1) - 1, 2 ** depth - 1)
    layer = tree[rowindices]
    return layer, rowindices, maxdepth


def evaltree(X, tree):
    p = None
    # TODO
    return p


def buildtree(X, Xs, Xi, y, depth, featurecosts=None):
    """
     Perform gradient boosting: construct trees to minimize loss.
    :param X: Input points in the original space.
    :param Xs: X values sorted ascendant.
    :param Xi: The indices that sort the input elements of X.
    :param y: The negative of the gradient loss function. See equation (10) of the paper.
    :param depth: Max-depth of the tree
    :param featurecosts: Cost associated to each feature to be used in preprocesslayer function
    :return: outputlabels, p
      - outputlabels: The predictions of the tree into pp
      - p: Prediction for each instance into p
    """
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

    # Checks that the row dimensions of X
    if RowsX != RowsXs:
        raise Exception('Row dimensions in X do not match row dimensions in Xs, Xi, or y')

    #  initialize with each instance at the root node
    n = np.zeros(n_samples, dtype=int)

    # initialize tree and compute default label
    defaultlabel = np.mean(y, axis=0)

    # initialize the root node with default prediction. The format is [feature, split, impurity]
    tree = np.append(np.full(3, -1.0), defaultlabel)[np.newaxis]

    # build tree layer-wise
    for d in range(depth - 1):
        # get parent nodes
        parents = getlayer(tree, (d + 1))[0]

        # prepare data for preprocessing
        data = DataObject(X.shape[1], y, n, (d + 1), tree)

        # compute preprocessed arguments
        m_infty, l_infty, arg_parents, arg_featurecosts = preprocesslayer_sqrimpurity(data, featurecosts)
        f = np.arange(0, numfeatures)
        # Splits says for each feature the best split points, impurity the sqrimpurity level
        # associated with that split-point and labels the prediction of the leaf node.
        splits, impurity, labels = buildlayer_sqrimpurity(Xs, Xi, y.T, n, f, m_infty, l_infty, arg_parents,
                                                          arg_featurecosts, reshape_labels=True)
        # pick best splits for each node
        bestfeatures = np.argmin(impurity, axis=0)
        best_indices = np.ravel_multi_index([bestfeatures, np.arange(len(bestfeatures))], impurity.shape)
        if len(best_indices) == 1:
            best_indices = best_indices[0]
        bestimpurity, bestsplits = np.take(impurity, best_indices),  np.take(splits, best_indices)

        # record splits for parent nodes
        _, pi, _ = getlayer(tree, d + 1)
        tree[pi, 0:3] = np.vstack([bestfeatures, bestsplits, bestimpurity]).T

        # record labels for child nodes
        children = np.full((parents.shape[0] * 2, parents.shape[1]), -1.0)
        bfpairs = np.tile(np.array([bestfeatures]).T, 2 * (parents.shape[1] - 3)).flatten()
        pairedindices = np.ravel_multi_index([bfpairs, np.arange(len(bfpairs))], labels.shape)
        selected_labels = np.take(labels, pairedindices).reshape(children[:, 3:].shape)
        children[:, 3:] = selected_labels
        tree = np.vstack([tree, children])

        # Update nodes
        Fv = bestfeatures[n]
        Sv = bestsplits[n] if isinstance(bestsplits, np.ndarray) else bestsplits
        Vv = np.take(X, np.ravel_multi_index([np.arange(X.shape[0]), Fv], X.shape))
        n = (2 * (n + 1) - np.squeeze(Vv < Sv)) - 1

    # update predictions
    leafnodes = getlayer(tree, depth)[0]  # Stores the last Layer of tree into LL
    outputlabels = leafnodes[:, 3:]  # Stores the predictions of the tree into pp
    p = outputlabels[n]  # Stores the prediction for each instance into p

    return tree, p


def evalensemble(X, ensemble, p):
    p = None
    # TODO
    return p

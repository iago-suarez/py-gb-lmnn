from unittest import TestCase

from scipy.io import loadmat

from main import knnclassifytreeomp
from mtrees import MTree


class TestTreeInfo(TestCase):

    def test_fit(self):
        MTree(None, None, None, None, None, None, None).fit(None, None)

    def test_predict(self):
        MTree(None, None, None, None, None, None, None).predict(None)

    def test_build(self):
        _, _, _, _, xTr, _, _, _, _ = loadmat('../data/segment.mat').values()
        treesize = 15

        tree = MTree.build(xTr, treesize)
        n_nodes, n_samples = tree.n_nodes, xTr.shape[1]
        self.assertEqual(tree.index.shape, (n_samples,))
        self.assertEqual(tree.pivots_x.shape, (19, n_nodes))
        self.assertEqual(tree.radius.shape, (n_nodes,))
        self.assertEqual(tree.jumpindex.shape, (2, n_nodes))
        self.assertEqual(tree.kids.shape, (2, n_nodes))

    def test_knnclassifytreeomp(self):
        _, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('../data/segment.mat').values()
        _, _, _, index, piv, radius, jumpindex, kids = loadmat('../data/my_tree.mat').values()
        n_nodes = len(radius.flatten())
        tree = MTree(index.flatten() - 1, piv, radius.flatten(), jumpindex - 1, kids, n_nodes, 15)
        err = knnclassifytreeomp([], xTr, yTr.astype(int), xTe, yTe.astype(int), 1, tree=tree)

        print("1-NN Error: " + str(err) + "%")

from unittest import TestCase

from scipy.io import loadmat

from mtrees import MTree


class TestTreeInfo(TestCase):

    def test_fit(self):
        MTree(None, None, None, None, None).fit(None, None)

    def test_predict(self):
        MTree(None, None, None, None, None).predict(None)

    def test_build(self):
        _, _, _, _, xTr, _, _, _, _ = loadmat('../data/segment.mat').values()
        treesize = 15

        tree = buildmtreemex(xTr, treesize)
        n_nodes, n_samples = tree.n_nodes, xTr.shape[1]
        self.assertEqual(tree.index.shape, (n_samples,))
        self.assertEqual(tree.pivots_x.shape, (19, n_nodes))
        self.assertEqual(tree.radius.shape, (n_nodes,))
        self.assertEqual(tree.jumpindex.shape, (2, n_nodes))
        self.assertEqual(tree.kids.shape, (2, n_nodes))

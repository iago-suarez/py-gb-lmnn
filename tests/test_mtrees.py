from unittest import TestCase

from scipy.io import loadmat

from gb_lmnn import lmnnobj
from main import knnclassifytreeomp
from mtrees import MTree, buildtree, buildlayer_sqrimpurity_openmp_multi
import numpy as np


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
        tree = MTree(index.flatten() - 1, piv, radius.flatten(), jumpindex - 1, kids - 1, n_nodes, 15)
        yTr, yTe = yTr.astype(int).flatten(), yTe.astype(int).flatten()
        eval, details, _ = knnclassifytreeomp([], xTr, yTr, xTe, yTe, 1, tree=tree)
        expected_eval = np.array([[0.04473304], [0.04112554]])
        print(f"1-NN Error: {100 * eval[1][0]}%")
        self.assertTrue(np.allclose(expected_eval, eval))

    def test_usemtreemex(self):
        xtrain, xtest, k, piv, radius, jumpindex, kids = \
            list(loadmat('../data/usemtreemex_example.mat').values())[-1].flatten()

        _, _, _, expected_iknn, expected_dists = loadmat('../data/usemtreemex_result.mat').values()
        expected_iknn -= 1

        n_nodes = len(radius.flatten())
        index = np.arange(xtrain.shape[1])
        tree = MTree(index, piv, radius.flatten(), jumpindex - 1, kids - 1, n_nodes, 15)

        iknn, dists = tree.findknn(xtrain, xtest, int(k))

        self.assertTrue(np.allclose(expected_iknn, iknn))
        self.assertTrue(np.allclose(expected_dists, dists))

    def test_lmnnobj(self):
        elements = list(loadmat('../data/lmnnobj_example1.mat').values())[-1].flatten()
        expected_hinge, expected_grad, pred, targets_ind, active_ind = elements

        hinge, grad = lmnnobj(pred, targets_ind - 1, active_ind - 1)

        self.assertTrue(np.allclose(expected_hinge, hinge))
        self.assertTrue(np.allclose(expected_grad, grad))

    def test_build_tree(self):
        # TODO
        _, _, _, X, Xs, Xi, y, depth = loadmat('../data/buildtree1.mat').values()
        _, _, _, expected_tree, expected_p = loadmat('../data/buildtree1_result.mat').values()

        defaultlabel = None
        tree, p = buildtree(X, Xs, Xi - 1, y, int(depth), defaultlabel, buildlayer=buildlayer_sqrimpurity_openmp_multi)

        self.assertTrue(np.allclose(expected_tree, tree))
        self.assertTrue(np.allclose(expected_p, p))

    def test_buildlayer(self):
        Xs, Xi, y, n, f, m_infty, l_infty, parents_labels, feature_cost = \
            list(loadmat('../data/build_layer_inputs.mat').values())[-1].flatten()
        _, _, _, expected_splits, expected_impurity, expected_labels = \
            loadmat('../data/build_layer_outputs.mat').values()

        splits, impurity, labels = buildlayer_sqrimpurity_openmp_multi(
            Xs, Xi - 1, y, np.squeeze(n - 1), f - 1, m_infty, l_infty, parents_labels, feature_cost)

        self.assertTrue(np.allclose(expected_splits, splits))
        self.assertTrue(np.allclose(expected_impurity, impurity))
        self.assertTrue(np.allclose(expected_labels, labels))

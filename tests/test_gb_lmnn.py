import sys
from unittest import TestCase

from scipy.io import loadmat
from sklearn.tree import DecisionTreeRegressor

from gb_lmnn import lmnn_obj_loss, find_impostors
from main import knn_error_score
import numpy as np


class TestGBLMNN(TestCase):

    def test_knnclassifytreeomp(self):
        _, _, _, xTe, xTr, xVa, yTr, yTe, yVa = loadmat('../data/segment.mat').values()
        yTr, yTe = yTr.astype(int).flatten(), yTe.astype(int).flatten()
        tr_eval, te_eval = knn_error_score([], xTr.T, yTr, xTe.T, yTe, 1)
        expected_tr_eval, expected_te_eval = 0.04473304, 0.04112554
        print(f"1-NN Error: {100 * expected_te_eval}%")
        self.assertAlmostEqual(expected_tr_eval, tr_eval)
        self.assertAlmostEqual(expected_te_eval, te_eval)

    def test_findimpostors(self):
        data = loadmat('../data/findimpostors.mat')
        labels, pred, expected_impostors = data['labels'], data['pred'], data['active']
        impostors = find_impostors(pred.T, labels.flatten() - 1, 7, 50)
        self.assertTrue(np.allclose(expected_impostors - 1, impostors.T))

    def test_lmnnobj(self):
        elements = list(loadmat('../data/lmnnobj_example1.mat').values())[-1].flatten()
        expected_hinge, expected_grad, pred, targets_ind, active_ind = elements

        hinge, grad = lmnn_obj_loss(pred.T, targets_ind.T - 1, active_ind.T - 1)

        self.assertTrue(np.allclose(expected_hinge, hinge))
        self.assertTrue(np.allclose(expected_grad, grad.T))

    def _test_experiment_sklearn_tree_fig2(self):
        import matplotlib.pyplot as plt

        n_samples_per_lbl = 50
        initial_dist = 0.01
        problem_scale = 5
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / n_samples_per_lbl)
        middle_pts = np.array([np.cos(angles), np.sin(angles)])
        class1 = (1 - initial_dist) * middle_pts
        class2 = (1 + initial_dist) * middle_pts
        X = np.hstack([class1, class2])
        X = X.T
        X *= problem_scale
        y = np.array([0] * n_samples_per_lbl + [1] * n_samples_per_lbl)
        mask1 = y == 0
        mask2 = y == 1

        T1 = (np.arange(0, n_samples_per_lbl) + np.array([[-1], [+1]]))
        T1 = T1 % n_samples_per_lbl
        T2 = T1 + n_samples_per_lbl
        targets = np.hstack([T1, T2]).T

        I2 = (np.arange(0, n_samples_per_lbl) + np.array([[-2], [-1], [+1], [+2]]))
        I2 = I2 % n_samples_per_lbl
        I1 = I2 + n_samples_per_lbl
        impostors = np.hstack([I1, I2]).T

        lr = 0.01
        for i in range(300):
            hinge, grad = lmnn_obj_loss(X, targets, impostors)

            tree = DecisionTreeRegressor(max_depth=4)
            tree.fit(X, -grad)
            p = tree.predict(X)

            if i % 10 == 0:
                fig, [ax_true, ax_pred] = plt.subplots(1, 2, sharey=True)
                fig.suptitle("Iteration {}, loss: {:.2f}".format(i, hinge.mean()))
                fig.set_size_inches(12.0, 6.0)
                ax_true.set_title("True -gradient")
                ax_true.plot(X[mask1, 0], X[mask1, 1], 'ro')
                ax_true.plot(X[mask2, 0], X[mask2, 1], 'o', color='blue')

                for x, y, dx, dy in zip(X[mask1, 0], X[mask1, 1], -grad[mask1, 0], -grad[mask1, 1]):
                    ax_true.arrow(x, y, dx, dy, color='red', head_width=0.2, head_length=0.3)
                for x, y, dx, dy in zip(X[mask2, 0], X[mask2, 1], -grad[mask2, 0], -grad[mask2, 1]):
                    ax_true.arrow(x, y, dx, dy, color='blue', head_width=0.2, head_length=0.3)

                ax_true.grid()
                ax_true.set_aspect('equal')

                ################################################################

                ax_pred.set_title("Predicted gradient")
                ax_pred.plot(X[mask1, 0], X[mask1, 1], 'ro')
                ax_pred.plot(X[mask2, 0], X[mask2, 1], 'o', color='blue')

                for x, y, dx, dy in zip(X[mask1, 0], X[mask1, 1], p[mask1, 0], p[mask1, 1]):
                    ax_pred.arrow(x, y, dx, dy, color='red', head_width=0.2, head_length=0.3)
                for x, y, dx, dy in zip(X[mask2, 0], X[mask2, 1], p[mask2, 0], p[mask2, 1]):
                    ax_pred.arrow(x, y, dx, dy, color='blue', head_width=0.2, head_length=0.3)

                ax_pred.grid()
                ax_pred.set_aspect('equal')

                plt.show()

            # Update data
            X += lr * p

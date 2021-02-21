import unittest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from main.RidgeReg import RidgeReg


class RidgeRegTest(unittest.TestCase):
    alpha = 0.5
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-1, 0.2, 0.9, 2.1, 3.4, 4.2, 5.6, 6.5, 7.3])
    X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1),
                                                        y,
                                                        random_state=1512)
    boston = load_boston()
    boston_x_train, boston_x_test, boston_y_train, boston_y_test = train_test_split(boston.data,
                                                                                    boston.target,
                                                                                    random_state=1512)

    ridreg_simple = RidgeReg(X_train, y_train, alpha)
    ridreg_simple.fit()

    ridreg_boston = RidgeReg(boston_x_train, boston_y_train, alpha)
    ridreg_boston.fit()

    def test_fit(self):
        self.assertTrue(len(self.ridreg_simple.coeffs))
        self.assertTrue(len(self.ridreg_boston.coeffs))

    def test_predict(self):
        self.ridreg_simple.predict(self.X_test)
        self.assertTrue(len(self.ridreg_simple.y_pred))

        self.ridreg_boston.predict(self.boston_x_test)
        self.assertTrue(len(self.ridreg_boston.y_pred))

    def test_score(self):
        pred_copy_simple = np.copy(self.ridreg_simple.y_pred)
        pred_copy_boston = np.copy(self.ridreg_boston.y_pred)

        self.assertTrue(len(self.y_test))
        self.assertTrue(len(self.boston_y_test))

        self.assertEqual(pred_copy_simple.shape, self.y_test.shape)
        self.assertEqual(pred_copy_boston.shape, self.boston_y_test.shape)


if __name__ == '__main__':
    unittest.main()

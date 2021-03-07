import unittest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from main.RegressionModel import RegressionModel


class RegModelTest(unittest.TestCase):
    # Setup testing data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-1, 0.2, 0.9, 2.1, 3.4, 4.2, 5.6, 6.5, 7.3])
    X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1),
                                                        y,
                                                        random_state=1512)
    boston = load_boston()
    boston_x_train, boston_x_test, boston_y_train, boston_y_test = train_test_split(boston.data,
                                                                                    boston.target,
                                                                                    random_state=1512)
    # Setup Linear data
    regression_single = RegressionModel(X_train, y_train)
    regression_single.linear_fit()

    regression_boston = RegressionModel(boston_x_train, boston_y_train)
    regression_boston.linear_fit()

    # Setup Ridge data
    ridge_single = RegressionModel(X_train, y_train)
    ridge_single.ridge_fit(alpha=0.6)

    ridge_boston = RegressionModel(boston_x_train, boston_y_train)
    ridge_boston.ridge_fit(alpha=0.6)

    # Setup Lasso Data
    lasso_single = RegressionModel(X_train, y_train)
    lasso_single.lasso_fit(alpha=0.6)

    lasso_boston = RegressionModel(boston_x_train, boston_y_train)
    lasso_boston.lasso_fit(alpha=0.6)

    def test_all_fit(self):
        self.assertTrue(len(self.regression_single.coeffs))
        self.assertTrue(len(self.regression_boston.coeffs))

        self.assertTrue(len(self.ridge_single.coeffs))
        self.assertTrue(len(self.ridge_boston.coeffs))

        self.assertTrue(len(self.lasso_single.coeffs))
        self.assertTrue(len(self.lasso_boston.coeffs))

    def test_predict(self):
        """
        Only one model (standard or ridge or lasso) is enough as the previous test checks if the coeffs array
        is not empty
        """
        self.regression_single.predict(self.X_test)
        self.assertTrue(len(self.regression_single.y_pred))
        self.regression_boston.predict(self.boston_x_test)
        self.assertTrue(len(self.regression_boston.y_pred))

    def test_score(self):
        """
        Similarly, score calculation is the same across all models, so one is enough.
        Make a NumPy copy of prediction data (as y_test is a NumPy array) and check if their shapes are equal
        (otherwise score function will not work)
        """
        pred_copy_simple = np.copy(self.regression_single.y_pred)
        pred_copy_boston = np.copy(self.regression_boston.y_pred)

        self.assertEqual(pred_copy_simple.shape, self.y_test.shape)
        self.assertEqual(pred_copy_boston.shape, self.boston_y_test.shape)


if __name__ == '__main__':
    unittest.main()

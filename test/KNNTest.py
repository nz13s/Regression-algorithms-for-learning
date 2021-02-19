import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from main.KNN import KNN, most_common, AlgTypeError


class TestKNN(unittest.TestCase):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        random_state=1512)

    knn = KNN(X_train, y_train, 3, 'r')
    knn.fit(X_test[0])
    knn.predict(y_test)

    def test_most_common(self):
        self.assertEqual(most_common([1, 2, 3, 4, 5, 5, 6, 7]), 5)

    def test_fit(self):
        self.assertNotEqual(KNN.distance, [])

    def test_predict(self):
        self.assertNotEqual(KNN.y_pred, [])

    def test_alg_type_exception(self):
        newKNN = KNN(self.X_train, self.y_train, 3, 'x')
        self.assertRaises(AlgTypeError, newKNN.fit, self.X_test[0])


if __name__ == '__main__':
    unittest.main()

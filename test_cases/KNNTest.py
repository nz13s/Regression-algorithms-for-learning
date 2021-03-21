import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from main.KNN import KNN


class KNNTest(unittest.TestCase):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        random_state=1512)

    knn = KNN(X_train, y_train, 3)
    knn.fit(X_test[0])

    def test_fit(self):
        self.assertTrue(len(self.knn.distance))

    def test_predict(self):
        self.knn.predict(self.X_test)
        self.assertTrue(len(self.knn.y_pred))

    def test_score(self):
        pred_copy = np.copy(self.knn.y_pred)
        self.assertTrue(len(self.y_test))
        self.assertEqual(pred_copy.shape, self.y_test.shape)


if __name__ == '__main__':
    unittest.main()

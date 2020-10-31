from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from main.KNNClassifier import KNNClassifier


class KNNTest:
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris['data'],
                                                        iris['target'],
                                                        random_state=1512)
    k = 3
    y_pred = []
    for entry in X_test:
        current_label = KNNClassifier(X_train, y_train, entry, k)
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

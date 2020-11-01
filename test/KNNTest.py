from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from main.KNNClassifier import KNNClassifier


class TestIris:
    """Iris KNN implementation:"""
    iris = load_iris()

    # 1512 is my birthday in the DDMM format for a pseudo-random number generator as said in the assignment.
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


class TestIonSphere:
    """Ionosphere implementation"""
    X_ionosphere = np.genfromtxt("ionosphere.txt",
                                 delimiter=",",
                                 usecols=np.arange(34))
    y_ionosphere = np.genfromtxt("ionosphere.txt",
                                 delimiter=",",
                                 usecols=np.arange(34),
                                 dtype='int')

    # 1512 is my birthday in the DDMM format for a pseudo-random number generator as said in the assignment.
    X_train, X_test, y_train, y_test = train_test_split(X_ionosphere,
                                                        y_ionosphere,
                                                        random_state=1512)

    """
    Ionosphere is built different from Iris dataset.
    For Iris, X_train[0] called a single "tuple" and y_train[0] called its label.
    For IS, an array of tuples is called from X_train[0] so we have to include
    an index and a while loop.
    """

    print("--------------------")

    # 3 nearest neighbors
    k = 3
    y_pred_total = []
    for entry in range(len(X_test)):  # take the first array
        index = 0
        y_pred_this_array = []
        while index < len(X_test[entry]):
            current_label = KNNClassifier(X_train[entry], y_train[entry], X_test[entry][index], k)
            print("The predicted label is {}".format(current_label))
            y_pred_this_array.append(current_label)
            index += 1
        y_pred_total.append(y_pred_this_array)

    accuracy = np.mean(y_pred_total == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

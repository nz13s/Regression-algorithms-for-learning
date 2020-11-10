from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from main.KNNClassifier import KNNClassifier
from main.GraphPlot import plot_knn_classifier


class TestIris:
    """Iris KNN implementation:"""
    iris = load_iris()

    # 1512 is my birthday in the DDMM format for a pseudo-random number generator as said in the assignment.
    X_train, X_test, y_train, y_test = train_test_split(iris['data'],
                                                        iris['target'],
                                                        random_state=1512)

    # 3 nearest neighbors
    k = 3
    y_pred = []
    for entry in X_test:
        print("The test tuple is {}".format(entry))
        current_label = KNNClassifier(X_train, y_train, entry, k)
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

    # Graph test
    plot_knn_classifier(X_train, y_train, X_test, y_test)


class TestIonSphere:
    """Ionosphere implementation"""
    X_ionosphere = np.genfromtxt("ionosphere.txt",
                                 delimiter=",",
                                 usecols=np.arange(34))
    y_ionosphere = np.genfromtxt("ionosphere.txt",
                                 delimiter=",",
                                 usecols=34,
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
    y_pred = []
    for entry in X_test:
        print("The test tuple is {}".format(entry))
        current_label = KNNClassifier(X_train, y_train, entry, k)
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

    # Graph test
    plot_knn_classifier(X_train, y_train, X_test, y_test)

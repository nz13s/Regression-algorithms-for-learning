from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from main.KNN import KNN
from main.GraphPlot import plot_knn


class TestIris:
    """Iris KNN implementation:"""
    iris = load_iris()

    # 1512 is my birthday in the DDMM format for a pseudo-random number generator as said in the assignment.
    X_train, X_test, y_train, y_test = train_test_split(iris['data'],
                                                        iris['target'],
                                                        random_state=1512)

    # 3 nearest neighbors classifier
    k = 3
    y_pred = []
    for entry in X_test:
        print("The test tuple is {}".format(entry))
        current_label = KNN(X_train, y_train, entry, k, "c")
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

    # Graph test
    # plot_knn(X_train, y_train, X_test, y_test, "c")

    print("--------------------")

    # 3 nearest neighbors regressor
    k = 3
    y_pred = []
    for entry in X_test:
        print("The test tuple is {}".format(entry))
        current_label = KNN(X_train, y_train, entry, k, "r")
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

    print("--------------------")


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

    # 3 nearest neighbors classifier
    k = 3
    y_pred = []
    for entry in X_test:
        print("The test tuple is {}".format(entry))
        current_label = KNN(X_train, y_train, entry, k, "c")
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

    # Graph test
    # plot_knn_classifier(X_train, y_train, X_test, y_test)

    print("--------------------")

    # 3 nearest neighbors regressor
    k = 3
    y_pred = []
    for entry in X_test:
        print("The test tuple is {}".format(entry))
        current_label = KNN(X_train, y_train, entry, k, "r")
        print("The predicted label is {}".format(current_label))
        y_pred.append(current_label)

    accuracy = np.mean(y_pred == y_test)
    print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

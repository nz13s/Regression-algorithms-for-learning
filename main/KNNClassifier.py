import operator
import numpy as np


def most_common(labels):
    """
    Find the most common element in a list. Function from:
    https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    :param labels: a list of labels for the K neighbors found.
    :return: most common label to assign to a test set.
    """
    return max(set(labels), key=labels.count)


def KNNClassifier(X_train, y_train, testTuple, k):
    """
    This function computes the classification of a point using the K Nearest Neighbors algorithm.
    The idea is to run this algorithm against every point in the X_train set.
    :param X_train: a matrix of X values to be used as a training set.
    :param y_train: a matrix of labels for X_train values.
    :param testTuple: a tuple from X_test array to be tested, e.g. X_test[0] for Iris.
    :param k: number of nearest neighbors.
    :return: classification label of a point (or multiple points).
    """

    distance = []  # Array to store distances to nearest neighbors

    # Append distances, point and its label
    for data in range(len(X_train)):
        E_distance = np.linalg.norm(X_train[data] - testTuple)
        distance.append([X_train[data], E_distance, y_train[data]])

    # Sort distance
    """
    This sorts distances by actual distances, thus we use index of 1 corresponding to 
    distance[x][1] which is distance.
    """
    distance.sort(key=operator.itemgetter(1))  # REPLACE THIS LATER

    # Create the neighbors array
    neighbors = []
    for x in range(k):
        neighbors.append((distance[x][0], distance[x][2]))

    # Predict the label for this tuple
    labels_list = []
    for i in range(len(neighbors)):
        labels_list.append(neighbors[i][1])

    # Now, for the output:
    print("The test tuple is {}".format(testTuple))
    print("The nearest neighbors are:")
    for i in range(len(neighbors)):
        print(neighbors[i])
    return most_common(labels_list)

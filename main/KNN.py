import operator
import numpy as np


class AlgTypeError(Exception):
    """Raised when alg_type is not recognised"""

    def __init__(self, alg_type, message="Wrong argument type. Only accepts r or c as input."):
        self.alg_type = alg_type
        self.message = message
        super().__init__(self.message)


def most_common(labels):
    """
    Find the most common element in a list. Function from:
    https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    :param labels: a list of labels for the K neighbors found.
    :return: most common label to assign to a test set.
    """
    return max(set(labels), key=labels.count)


class KNN:
    distance = []  # Array to store distances to nearest neighbors
    y_pred = []  # Array for predictions

    def __init__(self, X_train, y_train, k, alg_type):
        """
        Initialise the KNN model with training data, number of neighbors and type of algorithm
        (regression or classification)
        :param X_train: training values
        :param y_train: training labels
        :param k: number of neighbors to find
        :param alg_type: a character 'r' or 'c'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.alg_type = alg_type

    def fit(self, testTuple):
        """
        Creates a list of nearest neighbours for one given data tuple
        :param testTuple: a chunk from the test dataset
        """
        # Append distances, point and its label
        for data in range(len(self.X_train)):
            E_distance = np.linalg.norm(self.X_train[data] - testTuple)
            self.distance.append([self.X_train[data], E_distance, self.y_train[data]])

        # Sort distance
        """
        This sorts distances by actual distances, thus we use index of 1 corresponding to 
        distance[x][1] which is distance.
        """
        self.distance.sort(key=operator.itemgetter(1))  # REPLACE THIS LATER

        # Create the neighbors array and predict the label for this tuple
        neighbors = []
        labels_list = []
        for x in range(self.k):
            neighbors.append((self.distance[x][0], self.distance[x][2]))
            labels_list.append(self.distance[x][2])

        if self.alg_type == "c":
            return most_common(labels_list)
        elif self.alg_type == "r":
            return round(np.average(labels_list))
        else:
            raise AlgTypeError(alg_type=self.alg_type)

    def predict(self, X_test):
        """
        For the entirety of X_test, run the fit() function to make a list of nearest neighbors for each entry of X_test.
        Add the formed list of predicted labels to a list of all predicted data.
        :param X_test: all of the data to be tested and predicted upon
        """
        # Do fit for all tuples and fill the predictions array
        for entry in X_test:
            current_label = self.fit(entry)
            self.y_pred.append(current_label)

    def score(self, y_test):
        """
        Return the accuracy score of the algorithm if we know what y_pred should be (i.e. if we have y_test)
        :param y_test: known labels for testing data
        :return: accuracy score
        """
        return 1 - np.mean(self.y_pred == y_test)

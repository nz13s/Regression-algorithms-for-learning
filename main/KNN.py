import operator

import numpy as np


class KNN:
    def __init__(self, X_train, y_train, k):
        """
        Initialise the KNN model with training data, number of neighbors and type of algorithm
        (regression or classification)
        :param X_train: training values
        :param y_train: training labels
        :param k: number of neighbors to find
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.y_pred = []  # Array for predictions

    def fit(self, testTuple):
        """
        Creates a list of nearest neighbours for one given data tuple
        :param testTuple: a single element from the test dataset
        """
        distance = []

        # Append distances, point and its label to distance array
        for data in range(len(self.X_train)):
            E_distance = np.linalg.norm(self.X_train[data] - testTuple)
            distance.append([self.X_train[data], E_distance, self.y_train[data]])

        """
        This sorts distances by actual distances, thus we use index of 1 corresponding to 
        distance[x][1] which is distance.
        """
        distance.sort(key=operator.itemgetter(1))

        labels_list = []
        for x in range(self.k):
            labels_list.append(distance[x][2])
        return round(np.mean(labels_list))

    def predict(self, X_test):
        """
        For the entirety of X_test, run the fit() function to make a list of nearest neighbors for each entry of X_test.
        Add the obtained label to a list of all predicted data.
        :param X_test: all of the data to be tested and predicted upon
        """
        pred = []
        for entry in X_test:
            current_label = self.fit(entry)
            pred.append(current_label)
        self.y_pred = np.copy(pred)
        return self.y_pred

    def score(self, y_test):
        """
        Find the R^2 score for this dataset.
        :param y_test: actual labels for X_test
        :return: R^2 score
        """
        TSS = np.sum((y_test - np.mean(y_test)) ** 2)
        RSS = np.sum((self.y_pred - y_test) ** 2)
        R2 = (TSS - RSS) / TSS
        return R2

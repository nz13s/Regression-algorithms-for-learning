import matplotlib.pyplot as plt
from random import randint
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main.KNN import KNN
from main.RegressionModel import RegressionModel


class GraphPlot:
    def __init__(self):
        pass

    @staticmethod
    def plot(X_train, y_train):
        """
        Plot the model results with mode selection for pure results, accuracy comparisons or overlap
        :param X_train: training set of X values
        :param y_train: training set of y labels
        """
        # Make array of colors
        colors = []
        for i in range(X_train.shape[1]):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        for x_row, y in zip(X_train, y_train):
            for i in range(len(x_row)):
                plt.scatter(x_row[i], y, color=colors[i])
        plt.ylabel("Y")
        plt.xlabel("Data")
        plt.show()

    @staticmethod
    def knn_accuracy_plot(X_train, y_train, X_test, y_test):
        k = 1
        X = []
        Y = []
        while k <= 3:
            this_k_knn = KNN(X_train, y_train, k=k)
            this_k_knn.predict(X_test)
            score = this_k_knn.score(y_test)
            print(k, score)
            X.append(int(k))
            Y.append(score)
            plt.scatter(k, score)
            print(this_k_knn.y_pred)
            k += 1
        plt.plot(X, Y)
        plt.xlabel("K neighbors")
        plt.ylabel("R scores")
        plt.show()


    @staticmethod
    def ridge_accuracy(X_train, y_train, X_test, y_test):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        scores = []
        for a in alpha_values:
            reg_model = RegressionModel(X_train, y_train)
            reg_model.ridge_fit(alpha=a)
            reg_model.predict(X_test)
            score = reg_model.score(y_test)
            scores.append(score)
            plt.scatter(a, score, label="Alpha value {}".format(a))
        plt.plot(alpha_values, scores)
        plt.xlabel("Alpha values")
        plt.ylabel("R2 scores")
        plt.show()

    @staticmethod
    def lasso_accuracy(X_train, y_train, X_test, y_test):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        scores = []
        for a in alpha_values:
            reg_model = RegressionModel(X_train, y_train)
            reg_model.lasso_fit(alpha=a)
            reg_model.predict(X_test)
            score = reg_model.score(y_test)
            scores.append(score)
            plt.scatter(a, score, label="Alpha value {}".format(a))
        plt.plot(alpha_values, scores)
        plt.xlabel("Alpha values")
        plt.ylabel("R2 scores")
        plt.show()


class Main:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        random_state=1512)
    myKNN = KNN(X_train, y_train, k=3)
    pred = myKNN.predict(X_test)
    myplot = GraphPlot()
    # myplot.plot(X_train, y_train)
    score = myKNN.score(y_test)
    print(score)

    myplot.knn_accuracy_plot(X_train, y_train, X_test, y_test)

    # boston = load_boston()
    # boston_X_train, boston_X_test, boston_y_train, boston_y_test = train_test_split(boston.data,
    #                                                                                 boston.target,
    #                                                                                 random_state=1512)
    # scaler = StandardScaler()
    # scaler.fit(boston_X_train)
    # boston_X_train_scaled = scaler.transform(boston_X_train)
    # boston_X_test_scaled = scaler.transform(boston_X_test)
    #
    # reg_model = RegressionModel(X_train=boston_X_train_scaled, y_train=boston_y_train)
    # reg_model.lasso_fit(alpha=0.8)
    # reg_model.predict(X_test=boston_X_test_scaled)
    # print(reg_model.score(boston_y_test))
    # print(reg_model.y_pred)
    # regPlot = GraphPlot()
    # regPlot.plot(X_train=boston_X_train_scaled, y_train=boston_y_train)

    # regPlot.lasso_accuracy(boston_X_train_scaled, boston_y_train, boston_X_test_scaled, boston_y_test)

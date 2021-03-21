import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main.KNN import KNN
from main.RegressionModel import RegressionModel


class GraphPlot:
    def __init__(self, model_input):
        self.model = model_input

    def single_plot(self, X_test, y_test):
        X = []
        Y1 = []
        Y2 = []
        for row, y1, y2 in zip(X_test, y_test, self.model.y_pred):
            for item in row:
                X.append(item)
                Y1.append(y1)
                Y2.append(y2)
            plt.scatter(X, Y1, color='black', s=5)
        """
        Before we computed the coefficients array to find y_pred.
        Now it is sort of a reverse process - get 2 coefficients (intercept and slope)
        for a line from the predicted data to plot a best fit line.
        """
        coeffs = np.polyfit(X, Y2, 1)
        m = coeffs[0]
        b = coeffs[1]
        plt.plot(X, np.dot(X, m) + b, color='red')
        plt.show()

    @staticmethod
    def knn_accuracy(X_train, y_train, X_test, y_test):
        k = 1
        X = []
        Y = []
        while k <= len(X_train):
            this_k_knn = KNN(X_train, y_train, k=k)
            this_k_knn.predict(X_test)
            score = this_k_knn.score(y_test)
            X.append(int(k))
            Y.append(score)
            plt.scatter(k, score, color='black', marker='x', s=5)
            k += 1
        plt.plot(X, Y, color='green')
        plt.xlabel("K neighbors")
        plt.ylabel("R scores")
        plt.show()

    def multi_plot(self, X_test, y_test):
        # Find number of subplots
        cols = X_test.shape[1]
        if cols % 2 == 0:  # if even
            """
            e.g. 12 columns = 12 plots
            12 / 2 = 6
            2 x 6 matrix
            """
            subplot_columns = int((cols / 2))
            subplot_rows = int((cols / subplot_columns))
        else:  # if odd
            """
            e.g. 13 columns = 14 plots
            14 / 2 = 7
            2 x 7 matrix
            """
            round_cols = cols + 1
            subplot_columns = int(round_cols / 2)
            subplot_rows = int(round_cols / subplot_columns)

        pred = self.model.y_pred
        fig, axarr = plt.subplots(nrows=subplot_rows, ncols=subplot_columns, sharey=True)
        fig.tight_layout()
        axarr = axarr.flatten()
        X_test_T = X_test.T
        for i, column in zip(range(len(axarr)), X_test_T):
            ax = axarr[i]
            X = []
            Y1 = []
            Y2 = []
            for item, y1, y2 in zip(column, y_test, pred):
                X.append(item)
                Y1.append(y1)
                Y2.append(y2)
            ax.scatter(X, Y1, color='black', s=5)
            """
            Before we computed the coefficients array to find y_pred.
            Now it is sort of a reverse process - get 2 coefficients (intercept and slope)
            for a line from the predicted data to plot a best fit line.
            """
            coeffs = np.polyfit(X, Y2, 1)
            m = coeffs[0]
            b = coeffs[1]
            ax.plot(X, np.dot(X, m) + b, color='red')
            ax.set_title("Feature {}".format(i + 1), fontsize=10)
        plt.show()

    # No function for LS accuracy as there is no obvious X parameter that can influence the R^2 score.
    # Possibly make a y_test against y_pred graph?

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
            plt.scatter(a, score, label="Alpha value {}".format(a), color='black', marker='x', s=5)
        plt.plot(alpha_values, scores, color='green')
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
            plt.scatter(a, score, label="Alpha value {}".format(a), color='black', marker='x', s=5)
        plt.plot(alpha_values, scores, color='green')
        plt.xlabel("Alpha values")
        plt.ylabel("R2 scores")
        plt.show()


class Main:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        random_state=1512)
    myKNN = KNN(X_train, y_train, k=3)
    knn_prediction = myKNN.predict(X_test)

    myplot = GraphPlot(myKNN)
    myplot.single_plot(X_test, y_test)
    myplot.multi_plot(X_test, y_test)
    # myplot.knn_accuracy(X_train, y_train, X_test, y_test)

    # --------------------
    boston = load_boston()
    boston_X_train, boston_X_test, boston_y_train, boston_y_test = train_test_split(boston.data,
                                                                                    boston.target,
                                                                                    random_state=1512)
    scaler = StandardScaler()
    scaler.fit(boston_X_train)
    boston_X_train_scaled = scaler.transform(boston_X_train)
    boston_X_test_scaled = scaler.transform(boston_X_test)

    lasso = RegressionModel(boston_X_train_scaled, boston_y_train)
    lasso.ridge_fit(alpha=0.8)
    lasso_prediction = lasso.predict(boston_X_test_scaled)

    regPlot = GraphPlot(lasso)
    regPlot.single_plot(boston_X_test_scaled, boston_y_test)
    regPlot.multi_plot(boston_X_test_scaled, boston_y_test)
    # regPlot.lasso_accuracy(boston_X_train_scaled, boston_y_train, boston_X_test_scaled, boston_y_test)

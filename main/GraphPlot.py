import matplotlib.pyplot as plt
import numpy as np

from main.KNN import KNN
from main.RegressionModel import RegressionModel


class GraphPlot:
    def __init__(self):
        self.best_k = 0
        self.best_knn_score = 0
        self.best_ridge_a = 0
        self.best_ridge_score = 0
        self.best_lasso_a = 0
        self.best_lasso_score = 0

    @staticmethod
    def single_plot(model, X_test, y_test):
        X = []
        Y1 = []
        Y2 = []
        fig = plt.figure()
        for row, y1, y2 in zip(X_test, y_test, model.y_pred):
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
        fig.savefig('single.png')

    @staticmethod
    def multi_plot(model, X_test, y_test):
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

        fig, axarr = plt.subplots(nrows=subplot_rows, ncols=subplot_columns, sharey=True)
        fig.tight_layout()
        axarr = axarr.flatten()
        X_test_T = X_test.T
        for i, column in zip(range(len(axarr)), X_test_T):
            ax = axarr[i]
            X = []
            Y1 = []
            Y2 = []
            for item, y1, y2 in zip(column, y_test, model.y_pred):
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
        fig.savefig('multi.png')

    def knn_accuracy(self, X_train, y_train, X_test, y_test, show=False):
        k = 1
        X = []
        Y = []
        fig = plt.figure()
        while k <= len(X_train):
            this_k_knn = KNN(X_train, y_train, k=k)
            this_k_knn.predict(X_test)
            score = this_k_knn.score(y_test)
            if score > self.best_knn_score and score != 1.0:
                self.best_knn_score = score
                self.best_k = k
            if show:
                X.append(int(k))
                Y.append(score)
                plt.scatter(k, score, color='black', s=20)
            k += 1
        if show:
            plt.plot(X, Y, color='red')
            plt.xlabel("K neighbors")
            plt.ylabel("R scores")
            plt.show()
            fig.savefig('knn_accuracy.png')

    # No function for LS accuracy as there is no obvious X parameter that can influence the R^2 score.
    # Possibly make a y_test against y_pred graph?

    def ridge_accuracy(self, X_train, y_train, X_test, y_test, show=False):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        scores = []
        fig = plt.figure()
        for a in alpha_values:
            reg_model = RegressionModel(X_train, y_train)
            reg_model.ridge_fit(alpha=a)
            reg_model.predict(X_test)
            score = reg_model.score(y_test)
            if score > self.best_ridge_score and score != 1.0:
                self.best_ridge_score = score
                self.best_ridge_a = a
            if show:
                scores.append(score)
                plt.scatter(a, score, label="Alpha value {}".format(a), color='black', s=20)
        if show:
            plt.plot(alpha_values, scores, color='red')
            plt.xlabel("Alpha values")
            plt.ylabel("R2 scores")
            plt.show()
            fig.savefig('ridge_accuracy.png')

    def lasso_accuracy(self, X_train, y_train, X_test, y_test, show=False):
        alpha_values = np.append(np.arange(0.0, 1.0, 0.01), 1.0)
        scores = []
        fig = plt.figure()
        for a in alpha_values:
            reg_model = RegressionModel(X_train, y_train)
            reg_model.lasso_fit(alpha=a)
            reg_model.predict(X_test)
            score = reg_model.score(y_test)
            if score > self.best_lasso_score and score != 1.0:
                self.best_lasso_score = score
                self.best_lasso_a = a
            if show:
                scores.append(score)
                plt.scatter(a, score, label="Alpha value {}".format(a), color='black', s=20)
        if show:
            plt.plot(alpha_values, scores, color='red')
            plt.xlabel("Alpha values")
            plt.ylabel("R2 scores")
            plt.show()
            fig.savefig('lasso_accuracy.png')

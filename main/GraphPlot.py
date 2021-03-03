import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from main.KNN import KNN


class GraphPlot:
    def __init__(self, X_test):
        self.X_test = X_test
        self.KNN = KNN

    def plotKNN(self, mode, **kwargs):
        """
        Plot KNN results with mode selection for pure results, accuracy comparisons or overlap
        :param mode: "predict", "compare" or "overlap"
        :param kwargs: optional argument to include y_test if known to compare and see overlap
        """
        X = []
        X_overlap = []
        Y_test = []
        Y_pred = []
        overlap = []
        y_test = kwargs.get("y_test")
        if mode == "predict":
            for x_row, y in zip(self.X_test, self.KNN.y_pred):
                for item in x_row:
                    X.append(item)
                    Y_pred.append(y)
            plt.scatter(X, Y_pred, label="Prediction", c="green")

        if mode == "compare":
            for x_row, y1, y2 in zip(self.X_test, y_test, self.KNN.y_pred):
                for item in x_row:
                    X.append(item)
                    Y_test.append(y1)
                    Y_pred.append(y2)
            plt.scatter(X, Y_test, label="Original data", c="blue")
            plt.scatter(X, Y_pred, label="Prediction", c="green")

        if mode == "overlap":
            for x_row, y1, y2 in zip(self.X_test, y_test, self.KNN.y_pred):
                for item in x_row:
                    X.append(item)
                    Y_test.append(y1)
                    Y_pred.append(y2)
                    if y1 == y2:
                        overlap.append(y1)
                        X_overlap.append(item)
            plt.scatter(X, Y_test, label="Original data", c="blue")
            plt.scatter(X, Y_pred, label="Prediction", c="green")
            plt.scatter(X_overlap, overlap, label="Overlap", c="red")

        plt.legend()
        plt.ylabel("Y")
        plt.xlabel("Data")
        plt.show()


class Main:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        random_state=1512)
    myKNN = KNN(X_train, y_train, k=3, alg_type='r')
    myKNN.predict(X_test)
    myplot = GraphPlot(X_test)
    myplot.plotKNN(mode="overlap", y_test=y_test)
    score = myKNN.score(y_test)

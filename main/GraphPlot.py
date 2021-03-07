import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main.KNN import KNN
from main.RegressionModel import RegressionModel


class GraphPlot:
    def __init__(self, X_test, model_input):
        self.X_test = X_test
        self.model = model_input

    def plot(self, mode, **kwargs):
        """
        Plot the model results with mode selection for pure results, accuracy comparisons or overlap
        :param mode: "predict" or "compare"
        :param kwargs: optional argument to include y_test if known to compare and see overlap
        """
        X = []
        Y_test = []
        Y_pred = []
        y_test = kwargs.get("y_test")
        if mode == "predict":
            for x_row, y in zip(self.X_test, self.model.y_pred):
                for item in x_row:
                    X.append(item)
                    Y_pred.append(y)
            plt.scatter(X, Y_pred, label="Prediction", c="green")

        if mode == "compare":
            for x_row, y1, y2 in zip(self.X_test, y_test, self.model.y_pred):
                for item in x_row:
                    X.append(item)
                    Y_test.append(y1)
                    Y_pred.append(y2)
            plt.scatter(X, Y_test, label="Original data", c="blue")
            plt.scatter(X, Y_pred, label="Prediction", c="green")

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
    myplot = GraphPlot(X_test, myKNN)
    myplot.plot(mode="compare", y_test=y_test)
    score = myKNN.score(y_test)

    boston = load_boston()
    boston_X_train, boston_X_test, boston_y_train, boston_y_test = train_test_split(boston.data,
                                                                                    boston.target,
                                                                                    random_state=1512)
    scaler = StandardScaler()
    scaler.fit(boston_X_train)
    boston_X_train_scaled = scaler.transform(boston_X_train)
    boston_X_test_scaled = scaler.transform(boston_X_test)

    reg_model = RegressionModel(X_train=boston_X_train_scaled, y_train=boston_y_train)
    reg_model.linear_fit()
    reg_model.predict(boston_X_test_scaled)
    regPlot = GraphPlot(boston_X_test, reg_model)
    regPlot.plot(mode="compare", y_test=boston_y_test)

from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import numpy as np

from main.KNN import KNN
from main.GraphPlot import plot_knn
from main.LinReg import lin_reg
from main.RidgeReg import ridge_reg

# Test KNN:

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'],
                                                    iris['target'],
                                                    random_state=1512)

# Classification
k = 3
y_pred = []
for entry in X_test:
    print("The test tuple is {}".format(entry))
    current_label = KNN(X_train, y_train, entry, k, "c")
    print("The predicted label is {}".format(current_label))
    y_pred.append(current_label)

accuracy = np.mean(y_pred == y_test)
print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))
print("--------------------")

# Regression
k = 3
y_pred = []
for entry in X_test:
    print("The test tuple is {}".format(entry))
    current_label = KNN(X_train, y_train, entry, k, "r")
    print("The predicted label is {}".format(current_label))
    y_pred.append(current_label)

accuracy = np.mean(y_pred == y_test)
print("Error rate (rounded) is {}%".format(round((1 - accuracy) * 100)))

# Graph test
plot_knn(X_train, y_train, X_test, y_test, "c")
plot_knn(X_train, y_train, X_test, y_test, "r")

print("--------------------")

# Linear
print("Simple Linear Regression")
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([-1, 0.2, 0.9, 2.1, 3.4, 4.2, 5.6, 6.5, 7.3])
R = lin_reg(x, y)
print("My R2:", R)

# For simple LR, we need to reshape X here in order for TTT to work.
# We don't need to do it when setting up x as my lin_reg() reshapes it inside the function.
X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1),
                                                    y,
                                                    random_state=1512)
lr = LinearRegression().fit(X_train, y_train)
print("Scikit R2:", lr.score(X_test, y_test))

print("---")
print("Multiple Linear Regression")
d = load_boston()
R = lin_reg(data=d.data, target=d.target)
print("My R2:", R)
X_train, X_test, y_train, y_test = train_test_split(d.data,
                                                    d.target,
                                                    random_state=1512)
lr = LinearRegression().fit(X_train, y_train)
print("Scikit R2:", lr.score(X_test, y_test))

print("--------------------")
print("Ridge Regression")
for a in [0, 0.001, 0.01, 0.1, 1]:
    print("Current alpha:", a)
    ridge = Ridge(alpha=a).fit(X_train, y_train)
    R2 = ridge_reg(d.data, d.target, alpha=a)
    print("My R2:", R2)
    print("Scikit R2:", ridge.score(X_test, y_test))

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from main.RidgeReg import ridge_reg


class RidgeRegTest:
    # Testing simple LR
    print("Simple Ridge Regression")
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-1, 0.2, 0.9, 2.1, 3.4, 4.2, 5.6, 6.5, 7.3])
    R = ridge_reg(x, y, alpha=0.05)
    print("My R2:", R)

    # For simple LR, we need to reshape X here in order for TTT to work.
    # We don't need to do it when setting up x as my lin_reg() reshapes it inside the function.
    X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1),
                                                        y,
                                                        random_state=1512)
    lr = Ridge(alpha=0.05).fit(X_train, y_train)
    print("Scikit R2:", lr.score(X_test, y_test))

    print("---")

    print("Multiple Ridge regression")
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=1512)
    ridge = Ridge(alpha=0.05).fit(X_train, y_train)
    R2 = ridge_reg(boston.data, boston.target, alpha=0.05)
    print("My R2:", R2)
    print("Scikit R2:", ridge.score(X_test, y_test))

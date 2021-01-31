import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from main.LassoReg import lasso_reg


class LassoTest:
    alpha = 10

    # Testing simple LR
    print("Simple Lasso Regression")
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-1, 0.2, 0.9, 2.1, 3.4, 4.2, 5.6, 6.5, 7.3])
    R = lasso_reg(x, y, alpha=alpha)
    print("My R2:", R)

    # For simple LR, we need to reshape X here in order for TTT to work.
    # We don't need to do it when setting up x as my lin_reg() reshapes it inside the function.
    X_train, X_test, y_train, y_test = train_test_split(x.reshape(-1, 1),
                                                        y,
                                                        random_state=1512)
    lr = Lasso(alpha=alpha).fit(X_train, y_train)
    print("Scikit R2:", lr.score(X_test, y_test))

    print("---")

    print("Multiple Lasso regression")
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=1512)
    ridge = Lasso(alpha=alpha).fit(X_train, y_train)
    R2 = lasso_reg(boston.data, boston.target, alpha=alpha)
    print("My R2:", R2)
    print("Scikit R2:", ridge.score(X_test, y_test))

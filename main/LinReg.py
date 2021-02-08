import numpy as np
from sklearn.model_selection import train_test_split


def lin_reg(data, target):
    """
    Build a model for linear regression given a dataset of X and Y.

    1) For Simple Linear Regression:
    Coefficients for the line y = w*x + b:
    w (a.k.a b1) is the slope of the line.

              E(Xy)
    w =       -----
              E(x^2)

    b (a.k.a b0) is the intercept of the regression line (value of y when X = 0).

    2) For Multiple Linear Regression
    (formula for w taken from https://stattrek.com/multiple-regression/regression-coefficients.aspx):

    w = (X'X)^-1 X'Y

    Where b (the intercept b0) is the first element of w[].
    :param data: a matrix of X values.
    :param target: a matrix of labels for the data values.
    :return R2: accuracy score
    """
    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        random_state=1512)
    # Data size
    n = np.size(X_train)

    # If we have simple linear regression (has a single feature)...
    # find y = wx + b using cross-deviation and least squares.
    if len(X_train.shape) == 1:
        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        E_xy = 0
        E_x2 = 0
        for i in range(len(X_train)):
            E_xy += np.sum(X_train[i] * y_train[i] - n * X_mean * y_mean)  # cross-deviation xy
            E_x2 += np.sum(X_train[i] * X_train[i] - n * X_mean * y_mean)  # deviation around x

        # Using the formula, find w and b
        w = E_xy / E_x2
        b = y_mean - (w * X_mean)
        print("w = {}, b = {}".format(w, b))

        # Make predictions using the formula y = wx + b
        pred = []
        for i in range(len(X_test)):
            y_current = (w * X_test[i]) + b
            pred.append(y_current)

        y_pred = np.copy(pred)  # make a numpy array out of predictions
        # This will help to calculate RSS and TSS without a type error
        RSS = np.sum((y_pred - y_test) ** 2)
        TSS = np.sum((y_test - np.mean(y_test)) ** 2)
        R2 = (TSS - RSS) / TSS

    # If we have multiple linear regression...
    else:
        # Concatenate ones to X.
        ones = np.ones(shape=X_train.shape[0]).reshape(-1, 1)
        X_train = np.concatenate((ones, X_train), 1)

        '''
        This is needed for the formula for the matrical product to work, as:

        y = b0 * >1< + b1x1 + b2x2 + ... + bkxk

        Therefore, we need a column of ones (highlighted as >1<) in our X_train to satisfy the intercept b0
        at w[0].
        '''

        # Calculate coefficients from the formula in the introduction passage
        XT = np.transpose(X_train)
        XT_X_inv = np.linalg.inv(XT.dot(X_train))
        coeffs = XT_X_inv.dot(XT).dot(y_train)

        '''
        From the formula, we can find
        y = b0 + b1x1 + b2x2 + .... bkxk
        '''
        intercept = coeffs[0]  # first element is the intercept b0
        b_vals = coeffs[1:]  # the rest of coefficients starting from b[1]
        pred = []  # array for predictions
        print("w = {}, b = {}".format(b_vals, intercept))

        for entry in X_test:
            y_current = intercept  # start as y = b0 + ...
            for xi, bi in zip(entry, b_vals):
                y_current += bi * xi  # keep adding
            pred.append(y_current)

        y_pred = np.copy(pred)
        RSS = np.sum((y_pred - y_test) ** 2)
        TSS = np.sum((y_test - np.mean(y_test)) ** 2)
        R2 = (TSS - RSS) / TSS

    return R2

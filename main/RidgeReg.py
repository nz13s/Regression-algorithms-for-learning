import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


def ridge_reg(data, target, alpha):
    """
    Build a model for ridge regression given a dataset of X and Y.

    In linear regression, the formula for coefficients was w = (X' X)^-1 X' Y.
    In ridge regression, we add a value of alpha α to this formula
    (from https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf):

    w = (X'X + αI)^-1 X' Y

    where I is the identity matrix of X'X and b (the intercept b0) is the first element of w[].
    Otherwise, it operates in a similar way to linear regression with:
    y = b0 * 1 + b1x1+...+ bkxk for both methods (simple and multiple).
    :param data: a matrix of X values.
    :param target: a matrix of labels for the data values.
    :return R2: accuracy score
    """

    # Since we are using a uniform w formula, we don't need two methods for finding w and b.
    # Still, we need to reshape data if it only has one feature.
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        random_state=1512)

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
    XT_X = XT.dot(X_train)
    I = np.identity(len(XT_X))
    XT_X_aI = np.add(XT.dot(X_train), I.dot(alpha))
    XT_X_aI_inv = np.linalg.inv(XT_X_aI)
    coeffs = XT_X_aI_inv.dot(XT).dot(y_train)

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

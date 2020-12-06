import numpy as np
from sklearn.model_selection import train_test_split


def lasso_reg(data, target, alpha):
    """
    Build a model for LASSO regression given a dataset of X and Y.

    In ridge regression, we added a value of alpha α
    w = (X'X + αI)^-1 X' Y
    where I is the identity matrix of X'X and b (the intercept b0) is the first element of w[].

    From Chetan Patil's comment on https://stats.stackexchange.com/questions/176599/, we can see that:

    In ridge regression, for one w, w = xy / (x^2 + α). The denominator only becomes zero when α --> ∞.
    In LASSO regression, for one w, w = (2xy - α) / (2x^2). The numerator will become zero since we subtract α,
    making large values of w = 0.

    Therefore, I am going with the following formula (however I am fairly sure I made this up):

    w = (2X'X - αI)^-1 (2X' Y)

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
    #  Calculate coefficients from the formula in the introduction passage
    XT = np.transpose(X_train)
    XT_X = XT.dot(X_train)
    XT_Y = XT.dot(y_train)
    I = np.identity(len(XT_X))
    coeffs = np.linalg.inv(np.subtract(2*XT_X, I*alpha)).dot(2*XT_Y)

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

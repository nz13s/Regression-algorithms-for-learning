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
    making large values of w equal to 0.

    From "Lasso Regression" by Wessel van Wieringen
    (http://www.few.vu.nl/~wvanwie/Courses/HighdimensionalDataAnalysis/WNvanWieringen_HDDA_Lecture56_LassoRegression_20182019.pdf)
    I have found the following formula: X'X w = X'Y - 0.5 α ż, where ż(j) = sign{[w(α)]j}.
    Therefore, I believe the Lasso coefficient formula can be written as:

    w(a) = (X'X)^-1 (X'Y - 0.5αz)
    When w(a) < 0, z = -1, making the formula to be w = (X'X)^-1 (X'Y + 0.5α), bringing it closer to zero.
    When w(a) > 0, z = 1, making the formula to be w = (X'X)^-1 (X'Y - 0.5α), bringing it closer to zero.
    When w(a) = 0, z ∈ [-1,1] (which does not help to make this any more understandable)

    This is also evident in Patil's formula which will look similar if the numerator and denominator would be
    divided by 2. However, it is difficult to understand or code as it requires from us to know w and z simultaneously,
    which is not possible.

    A research from HSE (https://www.hse.ru/data/2018/03/15/1164355911/4.%20Ridge%20Regression%20and%20Lasso%20v1.pdf)
    presents an interesting and simple formula that builds on the Least Squares method:

    When w(LS) > 0, w(Lasso) = w(LS) - 0.5α.
    When w(LS) < 0, w(Lasso) = w(LS) + 0.5α.
    This formula combined will look like this:

    w(Lasso) = sign(w(LS)) (|w(LS)| - 0.5α)+

    The subscript (x)+ means that this bracket will be the maximum result of max(x, 0).

    If this does not work, I don't know what will.

    Like in Ridge, b (the intercept b0) is the first element of w[].
    Otherwise, it operates in a similar way to other linear regression problems with:
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
    # w(LS) = (X'X)^-1 X'Y
    w_least_squares = np.linalg.solve(np.transpose(X_train) @ X_train,
                                      np.transpose(X_train) @ y_train)

    # If matrix is orthogonal, then: w(Lasso) = sign(w(LS)) (|w(LS)| - 0.5α)+
    if np.transpose(X_train).dot(X_train) == np.identity(len(X_train)):
        coeffs = list(map(lambda w: np.sign(w) * max(abs(w) - (alpha / 2), 0), w_least_squares))

    # (X'X) * w(Lasso) = (X'Y - 0.5α * sign(w(LS))
    else:
        # we make a list of sign values of LS coefficients
        sign_list = []
        for w in w_least_squares:
            i = np.sign(w)  # -1, 1 or 0 if w=0
            sign_list.append(int(i))
        # solve using formula?
        signs = np.copy(sign_list)
        coeffs = np.linalg.solve(X_train.T @ X_train,
                                 X_train.T @ y_train - (alpha/2) * signs)

    for i, coef in enumerate(coeffs):
        if abs(coef) <= alpha/2:
            coeffs[i] = 0

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
    u = ((y_test - y_pred) ** 2).sum()
    v = ((y_test - y_test.mean()) ** 2).sum()
    R2 = 1 - (u / v)
    return R2

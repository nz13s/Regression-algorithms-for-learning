import numpy as np


class LinReg:
    coeffs = []
    y_pred = []

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        """
        Build a model for linear regression given a dataset of X and Y.

        In general:
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

        The good thing is that we can use the matrical formula above for both cases.

        Where b (the intercept b0) is the first element of w[].
        """

        ones = np.ones(shape=self.X_train.shape[0]).reshape(-1, 1)
        self.X_train = np.concatenate((ones, self.X_train), 1)

        '''
         ^ This is needed for the formula for the matrical product to work, as:

        y = b0 * >1< + b1x1 + b2x2 + ... + bkxk

        Therefore, we need a column of ones (highlighted as >1<) in our X_train to satisfy the intercept b0
        at w[0].
        '''

        # Calculate coefficients from the formula in the introduction passage
        self.coeffs = np.linalg.solve(np.transpose(self.X_train) @ self.X_train,
                                      np.transpose(self.X_train) @ self.y_train)

    def predict(self, X_test):
        intercept = self.coeffs[0]  # first element is the intercept b0
        b_vals = self.coeffs[1:]  # the rest of coefficients starting from b[1]
        pred = []  # array for predictions
        for entry in X_test:
            y_current = intercept  # start as y = b0 + ...
            for xi, bi in zip(entry, b_vals):
                y_current += bi * xi  # keep adding
            pred.append(y_current)
            self.y_pred = np.copy(pred)
        return self.y_pred

    def score(self, y_test):
        u = ((y_test - self.y_pred) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        R2 = 1 - (u / v)
        return R2

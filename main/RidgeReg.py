import numpy as np


class RidgeReg:
    coeffs = []
    y_pred = []

    def __init__(self, X_train, y_train, alpha):
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def fit(self):
        """
        Build a model for ridge regression given a dataset of X and Y.

        In linear regression, the formula for coefficients was w = (X' X)^-1 X' Y.
        In ridge regression, we add a value of alpha α to this formula
        (from https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf):

        w = (X'X + αI)^-1 X' Y

        where I is the identity matrix of X'X and b (the intercept b0) is the first element of w[].
        Otherwise, it operates in a similar way to linear regression with:
        y = b0 * 1 + b1x1+...+ bkxk for both methods (simple and multiple).
        """

        # Concatenate ones to X.
        ones = np.ones(shape=self.X_train.shape[0]).reshape(-1, 1)
        self.X_train = np.concatenate((ones, self.X_train), 1)

        '''
        ^ This is needed for the formula for the matrical product to work, as:
    
        y = b0 * >1< + b1x1 + b2x2 + ... + bkxk
    
        Therefore, we need a column of ones (highlighted as >1<) in our X_train to satisfy the intercept b0
        at w[0].
        '''

        # Calculate coefficients from the formula in the introduction passage
        # w = (X'X + αI)^-1 X' Y
        I = np.identity(len(np.transpose(self.X_train).dot(self.X_train)))
        self.coeffs = np.linalg.solve(np.transpose(self.X_train) @ self.X_train + I.dot(self.alpha),
                                      np.transpose(self.X_train) @ self.y_train)

    def predict(self, X_test):
        """
        From the formula, we can find
        y = b0 + b1x1 + b2x2 + .... bkxk
        """
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

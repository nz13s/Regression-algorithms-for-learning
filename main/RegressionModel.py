import numpy as np


class RegressionModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.coeffs = []
        self.y_pred = []

    def ls_fit(self):
        """
        Build a model for linear regression given a dataset of X and Y.
        (formula for w from https://stattrek.com/multiple-regression/regression-coefficients.aspx):

        w = (X'X)^-1 X'Y

        We can use the matrical formula above for both cases, where b (the intercept b0) is the first element of w[]
        and the rest are coefficients to use.
        """

        # Concatenate ones to X.
        ones = np.ones(shape=self.X_train.shape[0]).reshape(-1, 1)
        self.X_train = np.concatenate((ones, self.X_train), 1)

        '''
        The trick above is needed for the formula for the matrical product to work, as:

        y = b0 * >1< + b1x1 + b2x2 + ... + bkxk

        Therefore, we need a column of ones (highlighted as >1<) in our X_train to satisfy the intercept b0
        at w[0].
        '''

        # Calculate coefficients for each feature from the formula in the introduction passage
        self.coeffs = np.linalg.solve(np.transpose(self.X_train) @ self.X_train,
                                      np.transpose(self.X_train) @ self.y_train)

    def ridge_fit(self, alpha):
        """
        Build a model for ridge regression given a dataset of X and Y.

        In linear regression, the formula for coefficients was w = (X' X)^-1 X' Y.
        In ridge regression, we add a value of alpha α to this formula
        (from https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf):

        w = (X'X + αI)^-1 X' Y

        where I is the identity matrix of X'X and b (the intercept b0) is the first element of w[].
        Otherwise, it operates in a similar way to linear regression with:
        y = b0 * 1 + b1x1+...+ bkxk for both methods (simple and multiple).
        :param alpha: alpha value used as a regularisation parameter
        """

        # Concatenate ones to X.
        ones = np.ones(shape=self.X_train.shape[0]).reshape(-1, 1)
        self.X_train = np.concatenate((ones, self.X_train), 1)

        '''
        The trick above is needed for the formula for the matrical product to work, as:

        y = b0 * >1< + b1x1 + b2x2 + ... + bkxk

        Therefore, we need a column of ones (highlighted as >1<) in our X_train to satisfy the intercept b0
        at w[0].
        '''

        # Calculate coefficients from the formula in the introduction passage
        # w = (X'X + αI)^-1 X' Y
        I = np.identity(len(np.transpose(self.X_train).dot(self.X_train)))
        self.coeffs = np.linalg.solve(np.transpose(self.X_train) @ self.X_train + I.dot(alpha),
                                      np.transpose(self.X_train) @ self.y_train)

    def lasso_fit(self, alpha):
        """
        Build a model for LASSO regression given a dataset of X and Y.

        From "Lasso Regression" by Wessel van Wieringen
        (http://www.few.vu.nl/~wvanwie/Courses/HighdimensionalDataAnalysis/WNvanWieringen_HDDA_Lecture56_LassoRegression_20182019.pdf)
        and research from HSE (https://www.hse.ru/data/2018/03/15/1164355911/4.%20Ridge%20Regression%20and%20Lasso%20v1.pdf)
        I have combined the two formulas presented there for a single working solution.
        Assuming van Wieringen meant that the sign value he marked as ż came from w(LS):

        iff X'X = I (data is orthogonal)
            w(Lasso) = sign(w(LS)) (|w(LS)| - α/2)+
        otherwise
            w(Lasso) = (X'X)^-1 (X'Y - a/2 * sign(w(LS)) where w(Lasso)i = 0 if |w(Lasso)i| <= a/2

        The subscript (x)+ means that this bracket will be the maximum result of max(x, 0).

        Like in Ridge, b (the intercept b0) is the first element of w[].
        Otherwise, it operates in a similar way to other linear regression problems with:
        y = b0 * 1 + b1x1+...+ bkxk for both methods (simple and multiple).
        :param alpha: alpha value used as a regularisation parameter
        """

        # Concatenate ones to X.
        ones = np.ones(shape=self.X_train.shape[0]).reshape(-1, 1)
        self.X_train = np.concatenate((ones, self.X_train), 1)

        '''
        This is needed for the formula for the matrical product to work, as:

        y = b0 * >1< + b1x1 + b2x2 + ... + bkxk

        Therefore, we need a column of ones (highlighted as >1<) in our X_train to satisfy the intercept b0
        at w[0].
        '''

        #  Calculate coefficients from the formula in the introduction passage
        # w(LS) = (X'X)^-1 X'Y
        w_least_squares = np.linalg.solve(np.transpose(self.X_train) @ self.X_train,
                                          np.transpose(self.X_train) @ self.y_train)

        # If matrix is orthogonal, then: w(Lasso) = sign(w(LS)) (|w(LS)| - 0.5α)+
        if np.transpose(self.X_train).dot(self.X_train) == np.identity(len(self.X_train)):
            self.coeffs = list(map(lambda w: np.sign(w) * max(abs(w) - (alpha / 2), 0), w_least_squares))

        # (X'X) * w(Lasso) = (X'Y - 0.5α * sign(w(LS))
        else:
            # we make a list of sign values of LS coefficients
            sign_list = []
            for w in w_least_squares:
                i = np.sign(w)  # -1, 1 or 0 if w=0
                sign_list.append(int(i))
            signs = np.copy(sign_list)
            self.coeffs = np.linalg.solve(self.X_train.T @ self.X_train,
                                          self.X_train.T @ self.y_train - (alpha / 2) * signs)

            # Implement max(x, 0) the direct way
            for i, coef in enumerate(self.coeffs):
                if abs(coef) <= alpha / 2:
                    self.coeffs[i] = 0

    def predict(self, X_test):
        """
        Use the formula y = b0 + b1x1 + b2x2 + ... + bixi to build an array of predictions.
        The values of b may vary depening on the chosen algorithm, but the prediction method is the same across
        all of them.
        :param X_test: the data to make a prediction on
        :return: y_pred, array of predicted labels
        """
        intercept = self.coeffs[0]  # set first element as the intercept b0
        b_vals = self.coeffs[1:]  # set the rest of coefficients starting from b1
        pred = []  # array for predictions
        for entry in X_test:  # pick  row in X_test
            y_current = intercept  # start as y = b0 + ...
            for xi, bi in zip(entry, b_vals):  # each value of X in a row has its own coefficient in b_vals
                y_current += bi * xi  # find the product and add to current y value
            pred.append(y_current)
        self.y_pred = np.copy(pred)
        return self.y_pred

    def score(self, y_test):
        """
        Find the R^2 score for this dataset and algorithm.
        :param y_test: actual labels for X_test
        :return: R^2 score
        """
        TSS = np.sum((y_test - np.mean(y_test)) ** 2)
        RSS = np.sum((self.y_pred - y_test) ** 2)
        R2 = (TSS - RSS) / TSS
        return R2

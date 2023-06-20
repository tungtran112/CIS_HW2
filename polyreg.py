import numpy as np


class PolynomialRegression:
    def __init__(self, degree=1, regLambda=1E-8):
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None

    def polyfeatures(self, X, degree):
        n = X.shape[0]
        polyX = np.zeros((n, degree))

        for i in range(degree):
            polyX[:, i] = np.power(X, i + 1).flatten()

        return polyX

    def fit(self, X, y):
        X_poly = self.polyfeatures(X, self.degree)
        n, d = X_poly.shape
        X_poly = np.c_[np.ones((n, 1)), X_poly]  # Add bias term

        I = np.eye(d + 1)
        I[0, 0] = 0  # Exclude bias term from regularization
        self.theta = np.linalg.inv(X_poly.T.dot(X_poly) + self.regLambda * I).dot(X_poly.T).dot(y)

    def predict(self, X):
        X_poly = self.polyfeatures(X, self.degree)
        n, d = X_poly.shape
        X_poly = np.c_[np.ones((n, 1)), X_poly]  # Add bias term

        return X_poly.dot(self.theta)


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    for i in range(2, n):
        Xtrain_subset = Xtrain[:i+1]
        Ytrain_subset = Ytrain[:i+1]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset, Ytrain_subset)

        predictTrain = model.predict(Xtrain_subset)
        errorTrain[i] = np.mean((predictTrain - Ytrain_subset) ** 2)

        predictTest = model.predict(Xtest)
        errorTest[i] = np.mean((predictTest - Ytest) ** 2)

    return errorTrain, errorTest

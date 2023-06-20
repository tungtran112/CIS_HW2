import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        '''
        n = len(y)
        h = self.sigmoid(X.dot(theta))
        cost = -(1/n) * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
        regularization = (regLambda/(2*n)) * np.sum(np.square(theta[1:]))
        return cost + regularization

    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        '''
        n = len(y)
        h = self.sigmoid(X.dot(theta))
        gradient = (1/n) * (X.T.dot(h - y))
        regularization = (regLambda/n) * theta
        regularization[0] = 0  # Exclude regularization for bias term
        return gradient + regularization

    def fit(self, X, y):
        '''
        Trains the model
        '''
        n, d = X.shape
        X = np.c_[np.ones(n), X]  # Add bias term
        self.theta = np.zeros(d+1)

        for i in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            self.theta -= self.alpha * gradient

            if np.linalg.norm(gradient) < self.epsilon:
                break

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        '''
        n = X.shape[0]
        X = np.c_[np.ones(n), X]  # Add bias term
        return np.round(self.sigmoid(X.dot(self.theta)))

    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1 / (1 + np.exp(-Z))

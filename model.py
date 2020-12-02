import numpy as np




class LinearRegression:

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate

        self.iterations = iterations


    def fit(self, X, Y):
        self.m, self.n = X.shape

        self.weights = np.zeros(self.n)

        self.bias = 0

        self.X = X

        self.Y = Y

        for i in range(self.iterations):
            self.updateWeights()

        return self

    def updateWeights(self):
        yPred = self.predict(self.X)

        dW = - (2 * self.X.T.dot(self.Y - yPred)) / self.m

        db = - 2 * np.sum(self.Y - yPred) / self.m

        self.weights -= self.learning_rate * dW

        self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias


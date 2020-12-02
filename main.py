from model import LinearRegression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np


def readDatFile(fileName):
    datContent = [i.strip().split(',') for i in open(fileName).readlines()]
    datContent = [list(map(float, i)) for i in datContent]
    return datContent


def multiVariate():
    data = readDatFile("assets/data/multivariateData.dat")

    df = pd.DataFrame(data)

    X = df.iloc[:, :-1].values

    Y = df.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1 / 3, random_state=5)

    # Model training

    model = LinearRegression(iterations=1000, learning_rate=0.0000001)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Predicted values ", np.round(Y_pred, 2))

    print("Real values	 ", Y_test)

    print("Trained weights	 ", model.weights)

    print("Trained bias	 ", round(model.bias, 2))

def uniVariate():
    data = readDatFile("assets/data/univariateData.dat")
    df = pd.DataFrame(data)

    X = df.iloc[:, :-1].values

    Y = df.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1 / 3, random_state=5)

    # Model training

    model = LinearRegression(iterations=1000, learning_rate=0.001)

    model.fit(X_train, Y_train)

    # Prediction on test set

    Y_pred = model.predict(X_test)

    print("Predicted values ", np.round(Y_pred, 2))

    print("Real values	 ", Y_test)

    print("Trained weight	 ", model.weights)

    print("Trained bias	 ", round(model.bias, 2))

def main():

    print("Univariate Model")
    uniVariate()
    print("")
    print("=========================================================================")
    print("")
    print("Multivariate Model")
    multiVariate()


    # plt.scatter(X_test, Y_test, color='blue')
    #
    # plt.plot(X_test, Y_pred, color='orange')
    #
    # plt.title('Salary vs Experience')
    #
    # plt.xlabel('Years of Experience')
    #
    # plt.ylabel('Salary')
    #
    # plt.show()


if __name__ == "__main__":
    main()

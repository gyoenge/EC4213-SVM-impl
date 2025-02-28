import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy import random
from utils import *


def load_KSVM_dataset():
    # you can change noise and random_state where noise >= 0.15
    dataset = datasets.make_moons(
        n_samples=300, noise=0.3, random_state=20)
    X, y = dataset
    # 레이블을 -1, 1로 변환
    labels = np.unique(y)
    y = np.where(y == labels[1], 1, -1)
    return X, y


# KSVM Implement ####################################################################

class KSVM:
    def __init__(self, C=1.0, epoches=2000, learning_rate=0.001):
        self.C = C
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.kernel = None
        self.alpha = None
        self.b = None

    def _normalizeX(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y, kernel="linear"):
        # kernels init
        self.kernel = {
            "linear": lambda X1, X2: np.dot(X1, X2.T),
            # rbf : gamma = 1.0
            "rbf": lambda X1, X2: np.exp(-1*np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2),
            # polynomial : degree = 3
            "polynomial": lambda X1, X2: (1 + np.dot(X1, X2.T)) ** 3,
            # sigmoid : gamma = 1, coef0 = 1
            "sigmoid": lambda X1, X2: np.tanh(np.dot(X1, X2.T) + 1)
        }[kernel]

        # normalize X (fit 시에만)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = self._normalizeX(X)

        # init parameters
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.X_fit = X
        self.y_fit = y
        self.b = 0

        # Compute kernel matrix
        K = self.kernel(X, X)

        # optimize
        for _ in range(self.epoches):
            for i in range(n_samples):
                gradient = np.sum(self.alpha * y * K[:, i] * y[i]) - 1
                self.alpha[i] -= self.learning_rate * gradient / K[i, i]
                self.alpha[i] = np.clip(self.alpha[i], 0, self.C)

        # find Support Vectors
        self.support_vector_indices = np.where(
            (self.alpha > 1e-4) & (self.alpha < self.C))[0]
        self.b = np.mean([y[i] - np.sum(self.alpha * y * K[:, i])
                         for i in self.support_vector_indices])

    def predict(self, X):
        X = self._normalizeX(X)
        K = self.kernel(X, self.X_fit)
        output = np.dot(K, self.alpha * self.y_fit) + self.b
        return np.sign(output)  # predict as -1 or 1

    def print_result(self, X, y):
        # print accuracy
        y_pred = self.predict(X)
        acc = computeClassificationAcc(y, y_pred)
        print("KSVM Accuracy:", acc)
        # print misclassified
        # print_misclassified_points(X, y, y_pred)

    def plot_decision_boundary(self, X, y):
        plot_decision_boundary_nonlinear(
            self, X, y, title="KSVM with Support Vectors")


# KSVM Test ########################################################################
if __name__ == "__main__":
    X, y = load_KSVM_dataset()

    print("=========================KSVM, linear=========================")
    model = KSVM()
    model.fit(X, y, kernel='linear')
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")
    print("=========================KSVM, rbf=========================")
    model = KSVM()
    model.fit(X, y, kernel='rbf')
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")
    print("=========================KSVM, polynomial=========================")
    model = KSVM()
    model.fit(X, y, kernel='polynomial')
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")
    print("=========================KSVM, sigmoid=========================")
    model = KSVM()
    model.fit(X, y, kernel='sigmoid')
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")

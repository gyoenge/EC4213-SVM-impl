import numpy as np
from sklearn import datasets
from utils import *


def load_SSVM_dataset():
    iris = datasets.load_iris()
    X = iris.data[50:, 2:]
    y = iris.target[50:]
    # 레이블을 -1, 1로 변환
    labels = np.unique(y)
    y = np.where(y == labels[1], 1, -1)
    return X, y


# primal SSVM Implement ############################################################

class PrimalSSVM:
    def __init__(self, C=1.0, epoches=1000, learning_rate=0.01):
        # model weights & bias
        self.weights = None
        self.bias = None
        # learning hyperparameters
        self.epoches = epoches
        self.lr = learning_rate  # learning rate
        self.C = C

    def _normalizeX(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y, optimize="SGD"):
        if optimize == "SGD":
            self._SGD(X, y)
        elif optimize == "CGD":
            self._CGD(X, y)
        else:
            raise NameError(f"Unsupported optimization method: {optimize}")

    def _SGD(self, X, y):
        """Stochastic GD"""

        # normalize X (fit 시에만)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = self._normalizeX(X)

        n_samples, n_features = X.shape
        # init weights, bias with 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # optimize
        for epoch in range(self.epoches):
            for i in range(n_samples):
                condition = y[i] * \
                    (np.dot(X[i], self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * self.weights
                else:
                    self.weights -= self.lr * (
                        self.weights - self.C * np.dot(X[i], y[i])
                    )
                    self.bias += self.lr * self.C * y[i]

        # 가중치와 편향을 원래 스케일로 조정
        self.weights = self.weights / self.std
        self.bias = self.bias - np.sum(self.weights * self.mean)

        return self.weights, self.bias

    def _CGD(self, X, y):
        """Coordinate GD"""

        # normalize X (fit 시에만)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = self._normalizeX(X)

        n_samples, n_features = X.shape
        # init weights, bias with 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # optimize
        for epoch in range(self.epoches):
            for j in range(n_features):
                gradient_w = 0
                gradient_b = 0
                for i in range(n_samples):
                    condition = y[i] * \
                        (np.dot(X[i], self.weights) + self.bias) >= 1
                    if condition:
                        gradient_w += self.weights[j]
                    else:
                        gradient_w += self.weights[j] - self.C * X[i][j] * y[i]
                        gradient_b -= self.C * y[i]

                gradient_w /= n_samples
                gradient_b /= n_samples

                # 가중치와 바이어스 업데이트
                self.weights[j] -= self.lr * gradient_w
                self.bias -= self.lr * gradient_b

        # 가중치와 편향을 원래 스케일로 조정
        self.weights = self.weights / self.std
        self.bias = self.bias - np.sum(self.weights * self.mean)

        return self.weights, self.bias

    def predict(self, X, normalize=False):
        if normalize:
            X = self._normalizeX(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)  # predict as -1 or 1

    def print_result(self, X, y):
        # print weights, bias
        print("PrimalSSVM Weights:", self.weights, "Bias:", self.bias)
        # print accuracy
        y_pred = self.predict(X)
        acc = computeClassificationAcc(y, y_pred)
        print("PrimalSSVM Accuracy:", acc)
        # print misclassified
        # print_misclassified_points(X, y, y_pred)

    def plot_decision_boundary(self, X, y):
        y_pred = self.predict(X)
        # plot_decision_boundary_with_SV(
        #     X, y, self.weights, self.bias, "Primal SSVM with Support Vectors")
        plot_decision_boundary_with_Misclassified(
            X, y, self.weights, self.bias, y_pred, "Primal SSVM with Support Vectors & Misclassified")


# dual SSVM Implement ##############################################################
class DualSSVM:
    def __init__(self, C=1.0, epoches=1000, learning_rate=0.01):
        # model weights & bias
        self.weights = None
        self.bias = None
        # learning hyperparameters
        self.epoches = epoches
        self.lr = learning_rate  # learning rate
        self.C = C
        self.alpha = None

    def _normalizeX(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y):
        # normalize X (fit 시에만)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = self._normalizeX(X)

        n_samples, n_features = X.shape
        # init alpha, weights, bias with 0
        self.alpha = np.zeros(n_samples)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epoches):
            for i in range(n_samples):
                # Gradient of the dual objective with respect to alpha_i
                gradient = np.dot(self.weights, y[i] * X[i]) - 1

                # Update alpha_i using the projected gradient method
                alpha_i_old = self.alpha[i]
                # 0~C 사이 값으로 alpha제한
                alpha_i_new = np.clip(
                    alpha_i_old - gradient / (np.dot(X[i], X[i])), 0, self.C)
                self.alpha[i] = alpha_i_new

                # Update the weight vector
                self.weights += (alpha_i_new - alpha_i_old) * y[i] * X[i]

            # Calculate bias
            self.bias = np.mean(y - np.dot(X, self.weights))

        # 가중치와 편향을 원래 스케일로 조정
        self.weights = self.weights / self.std
        self.bias = self.bias - np.sum(self.weights * self.mean)

        return self.weights, self.bias

    def predict(self, X, normalize=False):
        if normalize:
            X = self._normalizeX(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)  # predict as -1 or 1

    def print_result(self, X, y):
        # print weights, bias
        print("DualSSVM Weights:", self.weights, "Bias:", self.bias)
        # print accuracy
        y_pred = self.predict(X)
        acc = computeClassificationAcc(y, y_pred)
        print("DualSSVM Accuracy:", acc)
        # print misclassified
        # print_misclassified_points(X, y, y_pred)

    def plot_decision_boundary(self, X, y):
        y_pred = self.predict(X)
        # plot_decision_boundary_with_SV(
        #     X, y, self.weights, self.bias, "Dual SSVM with Support Vectors")
        plot_decision_boundary_with_Misclassified(
            X, y, self.weights, self.bias, y_pred, "Dual SSVM with Support Vectors & Misclassified")


# SSVM Test ########################################################################
if __name__ == "__main__":
    X, y = load_SSVM_dataset()

    print("=========================PrimalSSVM, SGD=========================")
    model = PrimalSSVM()
    model.fit(X, y, optimize="SGD")
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")
    print("=========================PrimalSSVM, CGD=========================")
    model = PrimalSSVM()
    model.fit(X, y, optimize="CGD")
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")

    print("=========================DualSSVM=========================")
    model = DualSSVM()
    model.fit(X, y)
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")

import numpy as np
from sklearn import datasets
from utils import *

"""
Make sure you use the method given in the question!
"""


def load_HSVM_dataset():
    iris = datasets.load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    # 레이블을 -1, 1로 변환
    labels = np.unique(y)
    y = np.where(y == labels[1], 1, -1)
    return X, y


# HSVM Implement ####################################################################

def hinge_loss_gradient(X, y, weights, bias):
    n_samples, n_features = X.shape

    # init dw, db
    dw = np.zeros(n_features)
    db = 0

    # calculate gradient for given datapoints
    for i in range(n_samples):
        # check condition for sub-gradient with hinge loss
        condition = y[i] * (np.dot(weights, X[i]) + bias)
        if condition < 1:
            dw -= y[i] * X[i]
            db -= y[i]
        else:
            pass

    # normalize dw, db with n_samples
    dw /= n_samples
    db /= n_samples

    return dw, db


def train_svm(X, y, learning_rate=0.01, epochs=1000, C=1000, optimizer='CGD'):
    n_samples, n_features = X.shape

    # Normalize the features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # init weights, bias
    weights = np.zeros(n_features)
    bias = 0

    # 1) Stochastic sub-gradient descent (SGD)
    if optimizer == 'SGD':
        for epoch in range(epochs):
            # for i in range(n_samples):
            indices = np.random.permutation(n_samples)
            for i in indices:
                # Calculate the gradient for each sample
                dw, db = hinge_loss_gradient(X[i:i+1], y[i:i+1], weights, bias)

                # Update weights and bias with regularization term
                weights -= learning_rate * (C * dw + weights)
                bias -= learning_rate * C * db

    # 2) Coordinate sub-gradient descent (CGD)
    elif optimizer == 'CGD':
        for epoch in range(epochs):
            for j in range(n_features):
                gradient_w = 0
                gradient_b = 0
                # Calculate the gradient for each weight
                dw, db = hinge_loss_gradient(
                    X[:, j:j+1], y, weights[j:j+1], bias)

                # Update weights and bias with regularization term
                gradient_w += C * dw + weights[j:j+1]
                gradient_b += C * db

                # gradient_w /= n_samples
                # gradient_b /= n_samples

                # Update weights and bias
                weights[j] -= learning_rate * gradient_w
                bias -= learning_rate * gradient_b

                # Avoid overflow by capping the updates
                if np.abs(weights[j]) > 1e5:
                    weights[j] = np.sign(weights[j]) * 1e5
                if np.abs(bias) > 1e5:
                    bias = np.sign(bias) * 1e5

    # 3) Adam (ADAM)
    elif optimizer == 'ADAM':
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m_w = np.zeros(n_features)
        v_w = np.zeros(n_features)
        m_b = 0
        v_b = 0
        for epoch in range(epochs):
            dw, db = hinge_loss_gradient(X, y, weights, bias)

            # Update biased first moment estimate
            m_w = beta1 * m_w + (1 - beta1) * dw
            m_b = beta1 * m_b + (1 - beta1) * db

            # Update biased second raw moment estimate
            v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (db ** 2)

            # Compute bias-corrected first moment estimate
            m_w_hat = m_w / (1 - beta1 ** (epoch + 1))
            m_b_hat = m_b / (1 - beta1 ** (epoch + 1))

            # Compute bias-corrected second raw moment estimate
            v_w_hat = v_w / (1 - beta2 ** (epoch + 1))
            v_b_hat = v_b / (1 - beta2 ** (epoch + 1))

            # Update weights and bias
            weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    else:
        raise NameError(f"Unsupported optimization method: {optimizer}")

    # Denormalize the weights, bias
    weights /= std
    # bias -= np.sum(weights * mean / std)
    bias -= np.sum(weights * mean)

    return weights, bias


def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.sign(linear_output)


def print_result(X, y, weights, bias):
    # print weights, bias
    print("HSVM Weights:", weights, "Bias:", bias)
    # print accuracy
    y_pred = predict(X, weights, bias)
    acc = computeClassificationAcc(y, y_pred)
    print("HSVM Accuracy:", acc)
    # print misclassified
    # print_misclassified_points(X, y, y_pred)


# HSVM Test ########################################################################
if __name__ == "__main__":
    X, y = load_HSVM_dataset()

    # plot_decision_boundary(X, y, [5, -5], -10)

    print("=========================HSVM, SGD=========================")
    weights, bias = train_svm(X, y, learning_rate=0.01,
                              epochs=1000, C=1000, optimizer='SGD')
    print_result(X, y, weights, bias)
    y_pred = predict(X, weights, bias)
    plot_decision_boundary_with_Misclassified(
        X, y, weights, bias, y_pred, title="HSVM with Support Vectors & Misclassified")
    print("===========================================================")
    print("=========================HSVM, CGD=========================")
    weights, bias = train_svm(X, y, learning_rate=0.01,
                              epochs=1000, C=1000, optimizer='CGD')
    print_result(X, y, weights, bias)
    y_pred = predict(X, weights, bias)
    plot_decision_boundary_with_Misclassified(
        X, y, weights, bias, y_pred, title="HSVM with Support Vectors & Misclassified")
    print("===========================================================")
    print("=========================HSVM, ADAM=========================")
    weights, bias = train_svm(X, y, learning_rate=0.01,
                              epochs=1000, C=1000, optimizer='ADAM')
    print_result(X, y, weights, bias)
    y_pred = predict(X, weights, bias)
    plot_decision_boundary_with_Misclassified(
        X, y, weights, bias, y_pred, title="HSVM with Support Vectors & Misclassified")
    print("===========================================================")

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy import random
from utils import *
from SVM_hard import *
from SVM_soft import *
from SVM_kernel import *
from sklearn.svm import SVC  # discussion 결과 비교용


def print_dataset_info(X, y):
    # (optional) 데이터셋 확인용
    print("====================== check dataset ======================")
    print_feature_dimension(X)
    print_count_datapoints_all(X)
    print_count_datapoints_eachabel(y)
    plot_dataset(X, y)
    print("===========================================================")


############################################################################
###                              test for HSVM                           ###
############################################################################

def test_my_HSVM():
    # load dataset
    X, y = load_HSVM_dataset()

    # check dataset (optional)
    print_dataset_info(X, y)

    # HSVM, SGD
    print("=========================HSVM, SGD=========================")
    weights, bias = train_svm(X, y, learning_rate=0.01,
                              epochs=1000, C=1000, optimizer='SGD')
    print_result(X, y, weights, bias)
    y_pred = predict(X, weights, bias)
    plot_decision_boundary_with_Misclassified(
        X, y, weights, bias, y_pred, title="HSVM with Support Vectors & Misclassified")
    print("===========================================================")

    # HSVM, SGD
    print("=========================HSVM, CGD=========================")
    weights, bias = train_svm(X, y, learning_rate=0.01,
                              epochs=1000, C=1000, optimizer='CGD')
    print_result(X, y, weights, bias)
    y_pred = predict(X, weights, bias)
    plot_decision_boundary_with_Misclassified(
        X, y, weights, bias, y_pred, title="HSVM with Support Vectors & Misclassified")
    print("===========================================================")

    # HSVM, ADAM
    print("=========================HSVM, ADAM=========================")
    weights, bias = train_svm(X, y, learning_rate=0.01,
                              epochs=1000, C=1000, optimizer='ADAM')
    print_result(X, y, weights, bias)
    y_pred = predict(X, weights, bias)
    plot_decision_boundary_with_Misclassified(
        X, y, weights, bias, y_pred, title="HSVM with Support Vectors & Misclassified")
    print("===========================================================")


############################################################################
###                              test for SSVM                           ###
############################################################################

def test_my_SSVM():
    # load dataset
    X, y = load_SSVM_dataset()

    # check dataset (optional)
    print_dataset_info(X, y)

    # (1) Solve with Primal problem
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

    # (2) Solve with Dual problem
    print("=========================DualSSVM=========================")
    model = DualSSVM()
    model.fit(X, y)
    model.print_result(X, y)
    model.plot_decision_boundary(X, y)
    print("===========================================================")


############################################################################
###                              test for KSVM                           ###
############################################################################

def test_my_KSVM():
    # load dataset
    X, y = load_KSVM_dataset()

    # check dataset (optional)
    print_dataset_info(X, y)

    # KSVM
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


############################################################################
###                        compare with sklearn SVM                      ###
############################################################################
# sklearn library with same hyper-parameters

def fit_sklearn_SVMs(X, y, C, kernel, gamma=None, degree=None, coef0=None):
    if kernel == 'rbf':
        svm = SVC(C=C, kernel=kernel, gamma=gamma)
    elif kernel == 'poly':
        svm = SVC(C=C, kernel=kernel, degree=degree, coef0=coef0)
    elif kernel == 'sigmoid':
        svm = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0)
    else:
        # Linear or other kernel without additional parameters
        svm = SVC(C=C, kernel=kernel)
    svm.fit(X, y)
    return svm


def result_sklearn_SVMs(X, y, svm_model):
    y_pred = svm_model.predict(X)
    acc = computeClassificationAcc(y, y_pred)
    print("Accuracy: ", acc)


def test_sklearn_HSVM():
    # sklearn HSVM
    X, y = load_HSVM_dataset()
    sklearn_hsvm = fit_sklearn_SVMs(X, y, C=1000, kernel='linear')
    print("=========================sklearn, HSVM=========================")
    result_sklearn_SVMs(X, y, sklearn_hsvm)
    plot_decision_boundary_sklearn(
        sklearn_hsvm, X, y, "sklearn HSVM Decision Boundary")
    print("===============================================================")


def test_sklearn_SSVM():
    # sklearn SSVM
    X, y = load_SSVM_dataset()
    sklearn_ssvm = fit_sklearn_SVMs(X, y, C=1.0, kernel='linear')
    print("=========================sklearn, SSVM=========================")
    result_sklearn_SVMs(X, y, sklearn_ssvm)
    plot_decision_boundary_sklearn(
        sklearn_ssvm, X, y, "sklearn SSVM Decision Boundary")
    print("===============================================================")


def test_sklearn_KSVM():
    # sklearn KSVM
    X, y = load_KSVM_dataset()
    sklearn_ksvm = fit_sklearn_SVMs(X, y, C=1.0, kernel='rbf', gamma=1.0)
    print("=========================sklearn, KSVM, rbf=========================")
    result_sklearn_SVMs(X, y, sklearn_ksvm)
    plot_decision_boundary_sklearn(
        sklearn_ksvm, X, y, "sklearn KSVM rbf Decision Boundary")
    print("===============================================================")
    sklearn_ksvm = fit_sklearn_SVMs(
        X, y, C=1.0, kernel='poly', degree=3, coef0=1)
    print("=========================sklearn, KSVM, polynomial=========================")
    result_sklearn_SVMs(X, y, sklearn_ksvm)
    plot_decision_boundary_sklearn(
        sklearn_ksvm, X, y, "sklearn KSVM polynomial Decision Boundary")
    print("===============================================================")
    sklearn_ksvm = fit_sklearn_SVMs(
        X, y, C=1.0, kernel='sigmoid', gamma=1, coef0=1)
    print("=========================sklearn, KSVM, sigmoid=========================")
    result_sklearn_SVMs(X, y, sklearn_ksvm)
    plot_decision_boundary_sklearn(
        sklearn_ksvm, X, y, "sklearn KSVM sigmoid Decision Boundary")
    print("===============================================================")


### test main #############################################################
if __name__ == "__main__":
    # my HSVM
    test_my_HSVM()
    # my SSVM
    test_my_SSVM()
    # my KSVM
    test_my_KSVM()
    # sklearn SVMs
    test_sklearn_HSVM()
    test_sklearn_SSVM()
    test_sklearn_KSVM()

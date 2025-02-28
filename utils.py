import numpy as np
import matplotlib.pyplot as plt

# you can modify util functions here what you need
# this python file will not be included in the grading


#########################################################################################
# 기본 Accuracy compute & DecisionBoundary plot 관련 utils

def computeClassificationAcc(y_true, y_pred):
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def plot_decision_boundary(X, y, weights, bias, title="Decision Boundary"):
    # Determine unique classes and assign colors
    unique_classes = np.unique(y)
    colors = ['red' if label == unique_classes[0] else 'blue' for label in y]
    color_map = {unique_classes[0]: 'red', unique_classes[1]: 'blue'}

    # Scatter plot of the data points
    for class_value in unique_classes:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=color_map[class_value], label=f'Class {class_value}')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.dot(xy, weights) + bias
    Z = Z.reshape(XX.shape)

    # Legend and titles
    plt.title(title)
    plt.legend(loc='upper right')

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    plt.show()


def plot_decision_boundary_with_SV(X, y, weights, bias, title="Decision Boundary with Support Vectors"):
    # Determine unique classes and assign colors
    unique_classes = np.unique(y)
    # colors = ['red' if label == unique_classes[0] else 'blue' for label in y]
    color_map = {unique_classes[0]: 'red', unique_classes[1]: 'blue'}

    # Scatter plot of the data points
    for class_value in unique_classes:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=color_map[class_value], label=f'Class {class_value}')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.dot(xy, weights) + bias
    Z = Z.reshape(XX.shape)

    # Legend and titles
    plt.title(title)
    plt.legend(loc='upper right')

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Calculate the decision function values for all samples
    decision_values = y * (np.dot(X, weights) + bias)
    # Find indices where the decision values are close to 1 (on the margin)
    on_margin = np.isclose(decision_values, 1, atol=1e-4)
    support_vector_indices = np.where(on_margin)[0]

    # Highlight support vectors
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.show()


def plot_decision_boundary_with_Misclassified(X, y, weights, bias, y_pred, title="Decision Boundary with Misclassified Vectors"):
    # Determine unique classes and assign colors
    unique_classes = np.unique(y)
    colors = ['red' if label == unique_classes[0] else 'blue' for label in y]
    color_map = {unique_classes[0]: 'red', unique_classes[1]: 'blue'}

    # Scatter plot of the data points
    for class_value in unique_classes:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=color_map[class_value], label=f'Class {class_value}')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.dot(xy, weights) + bias
    Z = Z.reshape(XX.shape)

    # Legend and titles
    plt.title(title)
    plt.legend(loc='upper right')

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Calculate the decision function values for all samples
    decision_values = y * (np.dot(X, weights) + bias)
    # Find indices where the decision values are close to 1 (on the margin)
    on_margin = np.isclose(decision_values, 1, atol=1e-4)
    support_vector_indices = np.where(on_margin)[0]

    # Highlight support vectors
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')

    # Identify and highlight misclassified points
    misclassified_indices = np.where(y != y_pred)[0]
    for index in misclassified_indices:
        plt.scatter(X[index, 0], X[index, 1],
                    c=color_map[y[index]], )  # 한번더 점찍기
    plt.scatter(X[misclassified_indices, 0], X[misclassified_indices, 1],
                s=100, facecolors='none', edgecolors='yellow', linewidth=1, marker='o',
                label='Misclassified')

    plt.show()


def plot_decision_boundary_nonlinear(model, X, y, title="Non-linear Decision Boundary"):
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Normalize the grid
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on the grid
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Applying the same normalization as the model
    grid = (grid - model.mean) / model.std

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y,
                cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    # Plot support vectors
    support_vectors = X[model.support_vector_indices]
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.title(title)
    plt.show()


def plot_decision_boundary_sklearn(svm_model, X, y, title="sklearn SVMs Decision Boundary"):
    # 색상 지도 생성
    colors = ('red', 'blue')
    cmap = plt.cm.RdYlBu

    # 결정 경계를 그릴 그리드 생성
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))

    # SVM 모델을 사용하여 그리드 포인트의 클래스 레이블 예측
    Z = svm_model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # 결과 플롯
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 클래스 별로 데이터 포인트 플롯
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], label=cl)

    plt.title(title)
    plt.show()


#########################################################################################
# 학습 결과 관련 추가 utils

def print_misclassified_points(X, y, y_pred):
    # Find the indices where the actual and predicted labels do not match
    misclassified_indices = np.where(y != y_pred)[0]
    print("Misclassified points indices:", misclassified_indices)
    # Iterate over the indices and print the actual vs predicted values
    for index in misclassified_indices:
        print(f"Index {index}: Coordinates {X[index]}, Actual {
              y[index]}, Predicted {y_pred[index]}")


#########################################################################################
# 데이터셋 관련 utils

def plot_dataset(X, y, title="Dataset"):
    # Determine unique classes and assign colors
    unique_classes = np.unique(y)
    colors = ['red' if label == unique_classes[0] else 'blue' for label in y]
    color_map = {unique_classes[0]: 'red', unique_classes[1]: 'blue'}

    # Scatter plot of the data points
    for class_value in unique_classes:
        mask = y == class_value
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=color_map[class_value], label=f'Class {class_value}')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Legend and titles
    plt.title(title)
    plt.legend(loc='upper right')

    plt.show()


def print_feature_dimension(X):
    feature_dimension = X.shape[1]
    print(f"Data의 Feature dimension: {feature_dimension} dim")


def print_count_datapoints_all(X):
    count = X.shape[0]
    print(f"Datapoint 총 개수: {count}개")


def print_count_datapoints_eachabel(y):
    # 데이터셋에서 y 값이 변환된 후
    unique_labels, counts = np.unique(y, return_counts=True)

    # 각 레이블 별 데이터 개수 출력
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count}개")


def print_sample_datapoints(X, y):
    # 처음 3개 데이터와 레이블 출력
    print("First 3 samples from the iris dataset:")
    for i in range(3):
        print(f"Sample {i+1}: Features = {X[i]}, Target = {y[i]}")
    # 마지막 3개 데이터와 레이블 출력
    print("Last 3 samples from the iris dataset:")
    for i in range(-3, 0):
        print(f"Sample {100 + i + 1}: Features = {X[i]}, Target = {y[i]}")


def print_idx_datapoints(X, y, idx):
    print(f"Sample {idx}: Features = {X[idx]}, Target = {y[idx]}")


#########################################################################################
# util test
if __name__ == "__main__":
    # test for utils
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]

    y = np.where(y == 0, -1, 1)
    print_sample_datapoints(X, y)

    # n_samples, n_features = X.shape
    # weights = np.random.rand(n_features)
    # bias = np.random.rand()

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)

    # 가중치와 편향 출력
    weights = model.coef_[0]
    bias = model.intercept_[0]

    print(f"weights: {weights}, bias: {bias}")

    plot_decision_boundary(X, y, weights, bias)

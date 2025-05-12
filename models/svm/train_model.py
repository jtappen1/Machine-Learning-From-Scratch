import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from svm import SVM


def train_model():
    iris = load_iris()
    # Select only 2 features for simplification
    X = iris.data[:, :2]
    y = iris.target
    y = np.where(y == 0, -1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Iris Dataset (Setosa vs. Non-Setosa)')
    plt.show()

    svm = SVM(lr=0.001)
    svm.fit(X, y)

    def plot_decision_boundary(X, y, model):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = model.predict(xy).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
        plt.show()

    plot_decision_boundary(X, y, svm)


if __name__ == "__main__":
    train_model()
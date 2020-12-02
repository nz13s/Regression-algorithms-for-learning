import numpy as np
import matplotlib.pyplot as plt

from main.KNN import KNN


def plot_knn(X_train, y_train, X_test, y_test, type):
    X = []
    y = []
    k = 1
    print("Building model...")
    while k <= len(X_train):
        X.append(int(k))
        # Run KNN on all of X_test for current k
        y_pred_current = []
        for entry in X_test:
            current_label = KNN(X_train, y_train, entry, k, type)
            y_pred_current.append(current_label)
        accuracy_current = np.mean(y_pred_current == y_test)
        y.append(((1 - accuracy_current) * 100))
        k += 1

    x = np.copy(X)  # Need to make a NumPy array copy in order for parabola plot to work
    plt.plot(np.arange(len(x)) + 1, y, label='Error rates')
    plt.ylim(top=100)
    plt.xlabel("K neighbors")
    plt.ylabel("Error rate in %")
    plt.title("Error rate against model complexity")
    plt.legend()
    plt.show()

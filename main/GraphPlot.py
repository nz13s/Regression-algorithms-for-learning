import numpy as np
import matplotlib.pyplot as plt

from main.KNNClassifier import KNNClassifier


def plot_knn_classifier(X_train, y_train, X_test, y_test):
    X = []
    y = []
    k = 1
    print("Building model...")
    while k <= len(X_train):
        X.append(int(k))
        # Run KNN on all of X_test for current k
        y_pred_current = []
        for entry in X_test:
            current_label = KNNClassifier(X_train, y_train, entry, k)
            y_pred_current.append(current_label)
        accuracy_current = np.mean(y_pred_current == y_test)
        y.append(((1 - accuracy_current) * 100))
        k += 1
    print("Model constructed, parabola equation is:")
    x = np.copy(X)  # Need to make a NumPy array copy in order for parabola plot to work

    z = np.polyfit(x, y, 2)
    a = z[0].round(2)
    b = z[1].round(2)
    c = z[2].round(2)
    print("y = {}x^2 + {}x + {}".format(a, b, c))
    plt.plot(np.arange(len(x)) + 1, y, label='Error rates')
    plt.plot(x, (a * x ** 2) + b * x + c, 'r', label='Fitted curve')
    plt.ylim(top=100)
    plt.xlabel("K neighbors")
    plt.ylabel("Error rate in %")
    plt.title("Error rate against model complexity")
    plt.legend()
    plt.show()

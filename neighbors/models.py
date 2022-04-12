import numpy as np
from collections import Counter
from Utils.distances import  euclidean_distance

class KNeighborsClassifier:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()

    def predict(self, X):
        X = X.to_numpy()
        predicted_labels = np.array([self._predict(x) for x in X])
        return np.array(predicted_labels)

    def _predict(self, x):
        # Distances from each x in X_train
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # K-nearest
        k_idx = np.argsort(distances)[:self.k]

        # K_nearest labels
        k_labels = [self.y_train[i] for i in k_idx]

        # Voting mejority labels
        most_common = Counter(k_labels).most_common(1)

        return most_common[0][0]


class KNeigborsRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_label = [self._predict(x) for x in X]
        return np.array(predicted_label)

    def _predict(self, x):
        # Distance calculate
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # K-nearest indices
        k_idx = np.argsort(distances)[:self.k]

        # Nerest labels
        k_labels = [self.y_train[i] for i in k_idx]

        # Predicted label
        return np.array(k_labels).mean()
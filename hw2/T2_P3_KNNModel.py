from typing import List
import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x_pre in X_pred:
            dists = []
            for xi, yi in zip(self.X, self.y):
                tmp = self.dist(xi, x_pre)
                dists.append((yi, tmp))
            dists = sorted(dists, key= lambda tuple: tuple[1])
            y_pre = self.choose_y(dists)
            preds.append(y_pre)

        return np.array(preds)

    def choose_y(self, dists):
        y_list = []
        for i in range(self.K):
            y_list.append(dists[i][0])
        y = max(set(y_list), key=y_list.count)
        return y

    def dist(self, x1, x2):
        return np.sqrt(((x1[0] - x2[0]) / 3)**2 + (x1[1] - x2[1])**2)
    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y
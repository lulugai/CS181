from matplotlib.pyplot import axes, axis
import matplotlib.pyplot as plt
import numpy as np



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

def softmax(y):
    exp = np.exp(y - np.max(y))
    tmp = np.sum(exp)
    return exp / tmp

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.loss = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        N, D = X.shape

        #y -> one-hot
        tmps = np.unique(y)
        n_class = len(tmps)
        y1 = np.zeros((N, n_class))
        for i in range(n_class):
            for j in range(N):
                if y[j] == tmps[i]:
                    y1[j][i] = 1
        
        self.W = np.random.rand(D, n_class)
        for epoch in range(1000):
            y_h = X @ self.W
            tmp = []
            for j in range(N):
                y_hi = softmax(y_h[j].T).reshape(1,n_class)
                tmp.append(y_hi)
            y_h = np.concatenate(tmp, axis=0)
            # print(y_h.shape) #[N,C]

            for i in range(n_class):
                temploss = 0
                grad = 0
                for j in range(N):
                    temploss -= y1[j][i] * np.log(y_h[j][i])
                    grad += (y_h[j][i] - y1[j][i]) * X[j]
                grad /= N
                grad = grad + self.lam * self.W[:, i]
                self.W[:, i] -= self.eta * grad
            self.loss.append(temploss)
        # print('self.W', self.W.shape, X.shape)
        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        X_pred = np.concatenate((np.ones((X_pred.shape[0], 1)), X_pred), axis=1)#[N, D+1]
        y_h = X_pred @ self.W
        N, n_class = y_h.shape
        for i in range(y_h.shape[0]):
            y_hidx = softmax(y_h[i].T).reshape(n_class, 1).argmax()
            preds.append(y_hidx)
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.figure()
        plt.title("logistic loss over epochs")
        plt.xlabel('num epochs')
        plt.ylabel('loss')
        
        plt.plot(range(len(self.loss)), self.loss)
        plt.savefig(output_file + ".png")

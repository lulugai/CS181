#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

# print("y is:", y.shape)
# print(y)

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    # TODO: your code here
    W1 = alpha * np.array([[1., 0.], [0., 1.]])
    y_df = []
    for x_star in X:
        kerneln = []
        kernels = []    
        for x_n, y_n in zip(X, y):
            if x_star[0] == x_n[0] and x_star[1] == x_n[1]:
                continue
            residual = (x_n[:2] - x_star[:2]).reshape((1, -1))#1x2
            kernel = np.exp((-residual) @ W1 @ residual.T)
            kerneln.append(kernel * y_n)
            kernels.append(kernel)
        y_df.append(np.sum(kerneln) / np.sum(kernels)) 
    return np.array(y_df)

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    # TODO: your code here
    W1 = 10 * np.array([[1., 0.], [0., 1.]])
    y_df = []
    for x_star in X:
        kernels = []    
        for i, (x_n, y_n) in enumerate(zip(X, y)):
            if x_star[0] == x_n[0] and x_star[1] == x_n[1]:
                continue
            residual = (x_n[:2] - x_star[:2]).reshape((1, -1))#1x2
            kernel = np.exp((-residual) @ W1 @ residual.T)
            kernels.append((y_n, kernel))
        # print(kernels)
        kernels.sort(key= lambda tuple: tuple[1])
        # print('sort:', kernels)
        sum_y = []
        idx = 0
        while idx < k:
            dis = kernels[idx][0]
            sum_y.append(dis)
            idx += 1
        y_df.append(np.sum(sum_y) / k)
        # print('-----------------',sum_y)
    # print('y_pred:',np.array(y_df))
    return np.array(y_df)

def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()
    
# compute losses
def compute_lossKNN(k=1):
    ## TO DO
    loss = 0
    ypred = predict_knn(k)
    for i in range(len(ypred)):
        loss += (y[i]-ypred[i])**2
    return loss

# compute losses
def compute_lossKenal(alpha = 0.1):
    ## TO DO
    loss = 0
    ypred = predict_kernel(alpha)
    for i in range(len(ypred)):
        loss += (y[i]-ypred[i])**2
    return loss

for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    print(compute_lossKenal(alpha))
    plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    print(compute_lossKNN(k))
    plot_knn_preds(k)

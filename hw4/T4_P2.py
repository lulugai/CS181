# CS 181, Spring 2020
# Homework 4

import numpy as np
import matplotlib.pyplot as plt

# This line loads the images for you. Don't change it!
pics = np.load("data/images.npy", allow_pickle=False)

print(pics.shape)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

def l2_norm(x1, x2):
    res = 0
    for i in range(len(x1)):
        res += (x2[i] - x1[i])**2
    return res

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        # X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
        X = X.astype(float)
        K = self.K
        D = len(X[0])
        N = len(X)
        
        means = []
        # initlialize centroids
        indices = np.random.choice(range(len(X)), self.K, replace=False)
        for i in range (len(indices)):
            means.append(X[indices[i]])
        
        # objective function
        objective_vals = []
        # which group is this obs
        labels = [0]*N
        
        updated = True
        counter = 0
        # iterate until no longer update
        while updated:
            updated = False
            
            for i in range(N):
                # distance of i to each of the centroids
                dists = [l2_norm(X[i], means[j]) for j in range(K)]
                # closest centroid
                closest = np.argmin(dists)
                
                # if the label not the closest, update
                if labels[i] != closest:
                    labels[i] = closest
                    updated = True
            
            # bash out new centroids
            # counts per label
            counts = [0]*K
            # reset means
            for j in range(K):
                means[j] = np.array([0.0]*D)
            # bash
            for i in range(N):
                counts[labels[i]] += 1
                means[labels[i]] += X[i]
            for j in range(K):
                means[j] /= counts[j]
                
            # calculate objective function
            objective = 0
            for i in range(N):
                objective += l2_norm(X[i], means[labels[i]])
                
            objective_vals.append(objective)
            counter += 1
            print("step" + str(counter) + " done")
            
        self.labels = labels
        self.means = means
        self.objectives = objective_vals

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means

large_dataset = np.load("data/large_dataset.npy", allow_pickle=False)

K = 10
KMeansClassifier = KMeans(K=10)
# KMeansClassifier = KMeans(K=10, useKMeansPP=False)

print(large_dataset.shape)
KMeansClassifier.fit(large_dataset)

# plot
plt.plot(range(len(KMeansClassifier.objectives)), KMeansClassifier.objectives)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss over epoch for K=10 means classifier")
plt.show()

losses5 = []
losses10 = []
losses20 = []
for k in [5, 10, 20]:
    print("for K = " + str(k) + ": ")
    for rep in range(5):
        KMeansClassifier = KMeans(K=k)
        KMeansClassifier.fit(large_dataset)
        # checking "convergence"
        print(KMeansClassifier.objectives[-2]/KMeansClassifier.objectives[-1]>0.999)
        if k == 5:
            losses5.append(KMeansClassifier.objectives[-1])
        elif k == 10:
            losses10.append(KMeansClassifier.objectives[-1])
        elif k == 20:
            losses20.append(KMeansClassifier.objectives[-1])
    print()

losses5 = np.array(losses5)
losses10 = np.array(losses10)
losses20 = np.array(losses20)

mean_5 = np.mean(losses5)
mean_10 = np.mean(losses10)
mean_20 = np.mean(losses20)

std_5 = np.std(losses5)
std_10 = np.std(losses10)
std_20 = np.std(losses20)

types = ["K=5", "K=10", "K=20"]
x_pos = np.arange(3)
CTEs = [mean_5, mean_10, mean_20]
error = [std_5, std_10, std_20]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('loss (in e10)')
ax.set_xticks(x_pos)
ax.set_xticklabels(types)
ax.set_title('converged losses for values of K')
ax.yaxis.grid(True)
plt.show()

# This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
# plt.figure()
# plt.imshow(pics[0].reshape(28,28), cmap='Greys_r')
# plt.show()


class HAC(object):
	def __init__(self, linkage):
		self.linkage = linkage

# %%

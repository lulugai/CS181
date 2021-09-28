# Starter code for use with autograder.
import numpy as np
import matplotlib.pyplot as plt


def get_cumul_var(mnist_pics,
                  num_leading_components=500):

    """
    Perform PCA on mnist_pics and return cumulative fraction of variance
    explained by the leading k components.

    Returns:
        A (num_leading_components, ) numpy array where the ith component
        contains the cumulative fraction (between 0 and 1) of variance explained
        by the leading i components.

    Args:

        mnist_pics, (N x D) numpy array:
            Array containing MNIST images.  To pass the test case written in
            T5_P2_Autograder.py, mnist_pics must be a 2D (N x D) numpy array,
            where N is the number of examples, and D is the dimensionality of
            each example.

        num_leading_components, int:
            The variable representing k, the number of PCA components to use.
    """

    # TODO: compute PCA on input mnist_pics
    #center
    mnist_pics = mnist_pics - np.mean(mnist_pics, axis=0)

    N, D = mnist_pics.shape
    S = np.dot(mnist_pics.T, mnist_pics)/N
    u, s, vh = np.linalg.svd(S)
    # cols of u are eigenvectors then
    # since u orthogonal, w[i] = u.T[i]

    # TODO: return a (num_leading_components, ) numpy array with the cumulative
    # fraction of variance for the leading k components
    ret = np.zeros(num_leading_components)
    temp = np.sum(s)
    temp_sum = 0
    for i in range(num_leading_components):
        temp_sum += s[i]
        ret[i] = temp_sum / temp

    return ret

# Load MNIST.
mnist_pics = np.load("data/images.npy")

# Reshape mnist_pics to be a 2D numpy array.
num_images, height, width = mnist_pics.shape
mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

num_leading_components = 500

cum_var = get_cumul_var(
    mnist_pics=mnist_pics,
    num_leading_components=num_leading_components)

# Example of how to plot an image.
plt.figure()
plt.imshow(mnist_pics[0].reshape(28,28), cmap='Greys_r')
plt.show()

# p2.1
plt.plot(range(500), cum_var)
plt.title("cumulative variance explained over number of eigenvalues")
plt.xlabel("k largest eigenvalues")
plt.ylabel("cumulative variance explained")
plt.show()

#p2.2
mean_img = np.mean(mnist_pics, axis=0)
plt.imshow(mean_img.reshape(28, 28), cmap='Greys_r')
plt.show()

mnist_pics = mnist_pics - mean_img
N, D = mnist_pics.shape
S = np.dot(mnist_pics.T, mnist_pics)/N
u, s, vh = np.linalg.svd(S)
# cols of u are eigenvectors then
# since u orthogonal, w[i] = u.T[i]
ws = u.T[:10]
coefs = np.dot(mnist_pics, ws.T)
for i in range(len(ws)):
    plt.imshow(ws[i].reshape(28, 28), cmap='Greys_r')
    plt.savefig("pc" + str(i) + ".png")
    # plt.show()

# p2.3
mean_all = np.mean(mnist_pics, axis = 0)
mean_projections = np.array([np.dot(mean_all, ws[k])*ws[k] for k in range(len(ws))])
mean_rl = 0
for n in range(N):
    # get dist of each from mean, apparently
    mean_rl += np.linalg.norm(mean_all - mnist_pics[n])**2 #L2 **2

# pca rl
pca_rl = 0
for n in range(N):
    temp_project = np.array([np.dot(mnist_pics[n], ws[k])*ws[k] for k in range(len(ws))]) #[10,1]
    subspace_project = np.sum(temp_project, axis = 0)
    pca_rl += np.linalg.norm(mnist_pics[n] - subspace_project)**2                      

# pca_rl = np.linalg.norm(mean_all - np.sum(mean_projections, axis = 0))**2
print(pca_rl)
print(mean_rl)
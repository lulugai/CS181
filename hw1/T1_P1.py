import numpy as np

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])


def compute_loss(W):
    ## TO DO
    loss = 0
    np_data = np.array(data, dtype=np.float64)
    for x_star in np_data:
        kerneln = []
        kernels = []    
        for x_n in np_data:
            if x_star[0] == x_n[0] and x_star[1] == x_n[1]:
                continue
            residual = (x_n[:2] - x_star[:2]).reshape((1, -1))#1x2
            # print(residual,'vs', (x_n[:2] - x_star[:2]))
            kernel = np.exp((-residual) @ W @ residual.T)
            # print("kernel", kernel)
            kerneln.append(kernel * x_n[2])
            kernels.append(kernel)
        pred = np.sum(kerneln) / np.sum(kernels)
        loss += (pred - x_star[2])**2

    # loss /= np_data.shape[0]
    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))
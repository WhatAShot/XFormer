import numpy as np
import torch
import torch.nn.functional as F

def sampler_torch(x_train, y_train, n_samples=50):
    N, n_feat = x_train.shape
    n, = y_train.shape
    if N >= n_samples:
        idx = np.random.choice(N, n_samples, replace=False)
    else:
        raise ValueError('n_samples should be smaller than the number of training samples.')
    
    x_train = x_train[idx]
    y_train = y_train[idx]
    res = []
    for i in range(n_feat):
        res.append(torch.cat((x_train[:, i].reshape(-1, 1), y_train.reshape(-1, 1)), dim=1))

    return res

if __name__ == '__main__':

    x_train = np.random.randn(100, 8)
    y_train = np.random.randn(100)

    x_train_torch = torch.from_numpy(x_train)
    y_train_torch = torch.from_numpy(y_train)
    res = sampler_torch(x_train_torch, y_train_torch, n_samples=50)
    print(len(res))

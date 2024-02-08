import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.utils import get_spectral_rad, aug_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
import pickle 

def create_toy_data(T=10):
    N = 100
    l = 10
    num_class = T
    n = 10
    noise = 1

    X_list_all = []


    A = sp.coo_matrix((np.ones(10-1), (np.arange(10-1), np.arange(1, 10))), shape=(10, 10))
    A = A.T+A
    A_hat = aug_normalized_adjacency(A)
    A_hat = sparse_mx_to_torch_sparse_tensor(A_hat).float()
    A = sparse_mx_to_torch_sparse_tensor(A).float()
    A_list_orig = [A] * T
    A_list = [A_hat] * T

    for k in range(10):
        for i in range(10):
            X_list = []
            for j in range(10):
                X_temp = noise*np.random.rand(10, 10)
                if j == i:
                    X_temp[j][1] = 1
                else:
                    X_temp[j][1] = 0
                X_list.append(X_temp)
            X_list_all.append(X_list)
    
    label = np.repeat(np.arange(1, 11), 10)

    return A, A_hat, X_list_all, label

from sklearn.metrics import pairwise_distances
def dirichlet_energy(in_arr, mask_arr):
    dist_arr = pairwise_distances(in_arr, in_arr)
    dist_arr = dist_arr * dist_arr
    dist_arr = dist_arr * mask_arr
    return np.sqrt(dist_arr.sum(1).mean())


if __name__ == '__main__':
    A, A_hat, X_list_all, label = create_toy_data()
    print(A)
    print(A_hat)
    print(X_list_all)
    print(label)
    import pdb;pdb.set_trace()
    # save to file 
    with open('toy.pkl', 'wb') as f:
        pickle.dump([A, A_hat, X_list_all, label], f)
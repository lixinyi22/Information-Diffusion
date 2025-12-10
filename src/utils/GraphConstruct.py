import numpy as np
import torch
import scipy.sparse as sp

def LoadDiffusionGraph(all_cascade, user_size, window=10):
    user_cont = {}
    for i in range(user_size):
        user_cont[i] = []

    win = window
    for i in range(len(all_cascade)):
        cas = all_cascade[i]

        if len(cas)< win:
            for idx in cas:
                user_cont[idx] = list(set(user_cont[idx] + cas))
            continue
        for j in range(len(cas)-win+1):
            if (j+win) > len(cas):
                break
            cas_win = cas[j:j+win]
            for idx in cas_win:
                user_cont[idx] = list(set(user_cont[idx] + cas_win))

    indptr, indices, data = [], [], []
    indptr.append(0)
    idx = 0

    for j in user_cont.keys():

        # idx = source_users[j]
        if len(user_cont[j])==0:
            idx =  idx +1
            continue
        source = np.unique(user_cont[j])

        length = len(source)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(source[i])
            data.append(1)
            

    H_U = sp.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys())-idx, user_size))

    H_U_sum = 1.0 / H_U.sum(axis=1).reshape(1, -1)
    H_U_sum[H_U_sum == float("inf")] = 0

    # BH_T = H_S.T.multiply(1.0 / H_S.sum(axis=1).reshape(1, -1))
    BH_T = H_U.T.multiply(H_U_sum)
    BH_T = BH_T.T
    H = H_U.T

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    HG_User = np.dot(DH, BH_T).tocoo()

    HG_User = _convert_sp_mat_to_sp_tensor(HG_User)

    return HG_User

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
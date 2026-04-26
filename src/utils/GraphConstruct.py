import numpy as np
import torch
import scipy.sparse as sp
import utils.Constants as Constants


def BuildUserInfoParticipation(all_cascade, user_size, info_size):
    """Build a sparse (user_size, info_size) binary matrix from training cascades.

    Entry [u, i] = 1 means user u participated in cascade (info item) i.
    """
    rows, cols = [], []
    for cascade_idx in range(len(all_cascade)):
        for user_id in all_cascade[cascade_idx]:
            if user_id in (Constants.PAD, Constants.EOS):
                continue
            rows.append(user_id)
            cols.append(cascade_idx)

    rows = torch.LongTensor(rows)
    cols = torch.LongTensor(cols)
    values = torch.ones(len(rows), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (user_size, info_size)
    ).coalesce()
    adj = torch.sparse_coo_tensor(
        adj.indices(), torch.ones_like(adj.values()), adj.size()
    )
    return adj


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

    '''H_U_sum = 1.0 / H_U.sum(axis=1).reshape(1, -1)
    H_U_sum[H_U_sum == float("inf")] = 0'''
    H_U_row_sum = H_U.sum(axis=1).reshape(1, -1)
    H_U_sum = np.divide(1.0, H_U_row_sum, out=np.zeros_like(H_U_row_sum, dtype=float), where=(H_U_row_sum != 0))

    # BH_T = H_S.T.multiply(1.0 / H_S.sum(axis=1).reshape(1, -1))
    BH_T = H_U.T.multiply(H_U_sum)
    BH_T = BH_T.T
    H = H_U.T

    '''H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0'''
    H_row_sum = H.sum(axis=1).reshape(1, -1)
    H_sum = np.divide(1.0, H_row_sum, out=np.zeros_like(H_row_sum, dtype=float), where=(H_row_sum != 0))
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
    #return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)
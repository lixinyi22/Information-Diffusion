import os
import dhg
import torch
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ss

from collections import defaultdict
from torch_geometric.data import Data

class Options(object):
    
    def __init__(self, data_name = 'douban'):
        self.data = data_name+'/cascades.txt'
        self.u2idx_dict = data_name+'/u2idx.pickle'
        self.idx2u_dict = data_name+'/idx2u.pickle'
        self.save_path = ''
        self.net_data = data_name+'/edges.txt'
        self.embed_dim = 64
        self.train = data_name + '/cascade.txt'
        self.val = data_name + '/cascadevalid.txt'
        self.test = data_name + '/cascadetest.txt'

def Split_data(data_name):
	options = Options(data_name)
	u2idx = {}
	
	with open(options.u2idx_dict, 'rb') as handle:
		u2idx = pickle.load(handle)
	user_size = len(u2idx) + 1
		
	def build_dataset(dataset_path: str):
		t_cascades = []
		timestamps = []
		for line in open(dataset_path):
			if len(line.strip()) == 0:
				continue
			timestamplist = []
			userlist = []
			chunks = line.strip().split()
			for chunk in chunks:
				try:
					# Twitter,Douban
					if len(chunk.split(',')) == 2:
						user, timestamp = chunk.split(',')
					# Android,Christianity
					elif len(chunk.split(',')) == 3:
						root, user, timestamp = chunk.split(',')
						if root in u2idx:
							userlist.append(u2idx[root])
							timestamplist.append(float(timestamp))
					else:
						continue
				except:
					print(chunk)
				if user in u2idx:
					userlist.append(u2idx[user])
					timestamplist.append(float(timestamp))

			if len(userlist) >= 1 and len(userlist)<=510:
				t_cascades.append(userlist)
				timestamps.append(timestamplist)
			
		'''ordered by timestamps'''        
		order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x:x[1])]
		timestamps = sorted(timestamps)
		t_cascades[:] = [t_cascades[i] for i in order]
		cas_idx = [i for i in range(len(t_cascades))]
		return t_cascades, timestamps, cas_idx
	
	'''data split'''
	t_cascades, timestamps = [], []
	train, train_t, train_idx = build_dataset(options.train)
	t_cascades.extend(train)
	timestamps.extend(train_t)
	return user_size, t_cascades, timestamps

def LoadDynamicDiffusionGraph(path, time_step_split=8):
	t_cascades_pd = pd.read_csv(os.path.join(path, "diffusion_graph.csv"))

	t_cascades_pd = t_cascades_pd.sort_values(by="timestamp")
	t_cascades_length = t_cascades_pd.shape[0]
	step_length_x = t_cascades_length // time_step_split

	t_cascades_list = dict() 
	for x in range(step_length_x, t_cascades_length-step_length_x, step_length_x):
		t_cascades_pd_sub = t_cascades_pd[:x]
		t_cascades_sub_list = t_cascades_pd_sub.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
		sub_timesas = t_cascades_pd_sub["timestamp"].max()
			# t_cascades_list.append(t_cascades_sub_list)
		t_cascades_list[sub_timesas] = t_cascades_sub_list
		# + all
		t_cascades_sub_list = t_cascades_pd.apply(lambda x: (x["user1"], x["user2"]), axis=1).tolist()
		# t_cascades_list.append(t_cascades_sub_list) 
		sub_timesas = t_cascades_pd["timestamp"].max()
		t_cascades_list[sub_timesas] = t_cascades_sub_list

		dynamic_graph_dict_list = dict() 
		for key in sorted(t_cascades_list.keys()): 
			edges_list = t_cascades_list[key]
			# edges_list_tensor = torch.LongTensor(edges_list).t() 
			# loader = DataLoader(dataset, batch_size=32, shuffle=True) 
			# data = Data(edge_index=edges_list_tensor)
			cascade_dic = defaultdict(list)
			for upair in edges_list:
				cascade_dic[upair].append(1) 
			dynamic_graph_dict_list[key] = cascade_dic
		return dynamic_graph_dict_list 

def LoadDynamicHeteGraph(path):
	dy_diff_graph_list = LoadDynamicDiffusionGraph(path)
	dynamic_graph = dict()
	for x in sorted(dy_diff_graph_list.keys()):
		edges_list = []
		edges_type_list = []  # 0:follow relation,  1:repost relation
		edges_weight = []
		for key, value in dy_diff_graph_list[x].items():
			edges_list.append(key)
			edges_type_list.append(1)
			edges_weight.append(sum(value))

		edges_list_tensor = torch.LongTensor(edges_list).t()
		edges_type = torch.LongTensor(edges_type_list)
		edges_weight = torch.FloatTensor(edges_weight)

		data = Data(edge_index=edges_list_tensor, edge_type=edges_type, edge_weight=edges_weight)
		dynamic_graph[x] = data 
			
	return dynamic_graph

def ConHyperGraphList(graph_path, step_split=8):
	'''split the graph to sub graphs, return the list'''
	user_size, cascades, timestamps = Split_data(graph_path)

	e_size = len(cascades)+1
	n_size = user_size
	rows = []
	cols = []
	vals_time = []
	root_list = [0]

	for i in range(e_size-1):
		root_list.append(cascades[i][0])
		rows += cascades[i][:-1]
		cols +=[i+1]*(len(cascades[i])-1)
		#vals +=[1.0]*(len(cascades[i])-1)
		vals_time += timestamps[i][:-1]
		
	root_list = torch.tensor(root_list)
	Times = torch.sparse_coo_tensor(torch.Tensor([rows,cols]), torch.Tensor(vals_time), [n_size,e_size])
	times = Times.to_dense()
	zero_vec = torch.zeros_like(times)
	one_vec = torch.ones_like(times)

	time_sorted = []
	graph_list = {}

	for time in timestamps:
		time_sorted += time[:-1]
	time_sorted = sorted(time_sorted)
	split_length = len(time_sorted) // step_split

	for x in range(split_length, split_length * step_split , split_length):
		if x == split_length:
			sub_graph = torch.where(times > 0, one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
		else:
			sub_graph = torch.where(times > time_sorted[x-split_length], one_vec, zero_vec) - torch.where(times > time_sorted[x], one_vec, zero_vec)
			
		graph_list[time_sorted[x]] = sub_graph

	graphs = [graph_list, root_list]

	return graphs

def CascadeHypergraph(cascades, user_size, device):
	# cascades = cascades.tolist()
	edge_list = []
	for cascade in cascades:
		cascade = set(cascade)
		if len(cascade) > 2:
			cascade.discard(0)
		edge_list.append(cascade)

	cascade_hypergraph = dhg.Hypergraph(user_size, edge_list, device=device)

	return cascade_hypergraph

def DynamicCasHypergraph(graph_path, device, step_split=8):
	'''
	:param examples: 级联（用户）
	:param examples_times: 级联时间戳（用户参与级联的时间）
	:param user_size: 数据集中的所有用户
	:param device: 所在设备
	:param step_split: 划分几个超图
	:return: 超图序列
	'''
	user_size, examples, examples_times = Split_data(graph_path)
	hypergraph_list = []
	time_sorted = []
	for time in examples_times:
		time_sorted += time[:-1]
	time_sorted = sorted(time_sorted)   # 将所有时间戳升序排列
	split_length = len(time_sorted) // step_split    # 一个时间段包含的时间戳个数
	start_time = 0
	end_time = 0

	for x in range(split_length, split_length * step_split, split_length):
		# if x == split_length:
		#     end_time = time_sorted[x]
		# else:
		#     end_time = time_sorted[x]
		start_time = end_time
		end_time = time_sorted[x]

		selected_examples = []
		for i in range(len(examples)):
			example = examples[i]
			example_times = examples_times[i]
			if isinstance(example, list):
				example = torch.tensor(example)
				example_times = torch.tensor(example_times, dtype=torch.float64)
			selected_example = torch.where((example_times < end_time) & (example_times > start_time), example, torch.zeros_like(example))
			# print(selected_example)
			selected_examples.append(selected_example.numpy().tolist())

		sub_hypergraph = CascadeHypergraph(selected_examples, user_size, device=device)
		# print(sub_hypergraph)
		hypergraph_list.append(sub_hypergraph)

	# =============== 最后一张超图 ===============
	start_time = end_time
	selected_examples = []
	for i in range(len(examples)):
		example = examples[i]
		example_times = examples_times[i]
		if isinstance(example, list):
			example = torch.tensor(example)
			example_times = torch.tensor(example_times, dtype=torch.float64)
		selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))
		# print(selected_example)
		selected_examples.append(selected_example.numpy().tolist())
	hypergraph_list.append(CascadeHypergraph(selected_examples, user_size, device=device))

	return hypergraph_list

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def Read_all_cascade(path):
    with open(os.path.join(path, "u2idx.pickle"), 'rb') as handle:
        u2idx = pickle.load(handle)
    
    '''user size'''
    user_size = len(u2idx) + 1

    '''load train data, validation data and test data'''
    t_cascades = []
    timestamps = []

    ####process the raw data
    for line in open(os.path.join(path, "cascade.txt")):
        if len(line.strip()) == 0:
            continue

        timestamplist = []
        userlist = []
        chunks = line.strip().split()
        for i, chunk in enumerate(chunks):
            try:
                # Twitter,Douban
                if len(chunk.split(',')) == 2:
                    user, timestamp = chunk.split(',')
                # Android,Christianity
                elif len(chunk.split(',')) == 3:
                    root, user, timestamp = chunk.split(',')
                    if root in u2idx:
                        userlist.append(u2idx[root])
                        timestamplist.append(float(timestamp))
                else:
                    continue
            except:
                print(chunk)
            if user in u2idx:
                userlist.append(u2idx[user])
                timestamplist.append(float(timestamp))  

        if len(userlist) >= 1 and len(userlist)<=510:
            t_cascades.append(userlist)
            timestamps.append(timestamplist)

    return user_size, t_cascades

def ConHypergraph(path, window=10):

    user_size, all_cascade = Read_all_cascade(path)

    ###context
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
            

    H_U = ss.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys())-idx, user_size))

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

    '''U-I hypergraph'''
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j])

        length = len(items)

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))

    H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    H_T_sum[H_T_sum == float("inf")] = 0

    # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = H_T.T.multiply(H_T_sum)
    BH_T = BH_T.T
    H = H_T.T

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    HG_Item = np.dot(DH, BH_T).tocoo()


    HG_Item = _convert_sp_mat_to_sp_tensor(HG_Item)
    HG_User = _convert_sp_mat_to_sp_tensor(HG_User)

    return [HG_Item, HG_User]
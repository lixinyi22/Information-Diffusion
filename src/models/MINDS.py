# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from models.GraphEncoder import *
from models.TransformerBlock import TransformerBlock
from utils import layers

class MINDS(SequentialModel):
	extra_log_args = ['emb_size', 'num_layers', 'num_heads']
	reader = 'MINDSReader'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.')
		parser.add_argument('--num_heads', type=int, default=4,
							help='Number of attention heads.')
		return SequentialModel.parse_model_args(parser)	

	def __init__(self, args, corpus):
		SequentialModel.__init__(self, args, corpus)
		self.emb_size = args.emb_size
		self.max_his = args.history_max
		self.num_layers = args.num_layers
		self.num_heads = args.num_heads
		self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
		self._base_define_params(args)
		self.apply(self.init_weights)

	def _base_define_params(self, args):
		# additional
		self.dropout_layer = nn.Dropout(self.dropout)
		self.pos_embedding = nn.Embedding(self.max_his + 1, 8)
		self.linear = nn.Linear(self.emb_size + 8, self.emb_size)
		self.gnn_diffusion_layer = DynamicCasHGNN(self.item_num, self.emb_size, dropout=self.dropout)
		self.cascadeLSTM = nn.LSTM(self.emb_size, self.emb_size, num_layers=1, batch_first=True)
		
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
		self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
		self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
		self.his_fusion = Fusion(self.emb_size)
		self.item_fusion = Fusion(self.emb_size)
		self.transformer_block = nn.ModuleList([
			layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
									dropout=self.dropout, kq_same=False)
			for _ in range(self.num_layers)
		])

	def forward(self, feed_dict):
		self.check_list = []
		i_ids = feed_dict['item_id']  # [batch_size, -1]
		history = feed_dict['history_items']  # [batch_size, history_max]
		lengths = feed_dict['lengths']  # [batch_size]
		diffusion_graph = feed_dict['diffusion_graph']
		timestamp = feed_dict['history_times']
		batch_size, seq_len = history.shape
		graph_user_embedding = self.gnn_diffusion_layer(diffusion_graph)

		valid_his = (history > 0).long()
		mask = (history == 0)
		his_vectors = self.i_embeddings(history)

		# Position embedding
		# lengths:  [4, 2, 5]
		# position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
		position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
		pos_vectors = self.p_embeddings(position)
		his_vectors = his_vectors + pos_vectors

		# Self-attention
		causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=int))
		attn_mask = torch.from_numpy(causality_mask).to(self.device)
		# attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
		for block in self.transformer_block:
			his_vectors = block(his_vectors, attn_mask)
		his_vectors = his_vectors * valid_his[:, :, None].float()

		#his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
		his_vector = his_vectors.sum(1) / lengths[:, None].float()
		# ↑ average pooling is shown to be more effective than the most recent embedding
		

		# item_embedding fusion
		i_vectors = self.i_embeddings(i_ids)
		dy_i_vectors = F.embedding(i_ids, graph_user_embedding)
		i_vectors, i_gate = self.item_fusion(dy_i_vectors, i_vectors)
		
		# MINDS-local
		'''sender_cas_embedding, _ = self.cascadeLSTM(his_vectors)  # H^cas     (batch_size, 200, emb_dim)
		example_len = torch.count_nonzero(history, 1)  # 统计每个观察到的级联的长度，去掉用户0   (batch_size, 1)
		H_cas = []
		for i in range(batch_size):
			H_cas.append(sender_cas_embedding[i, example_len[i] - 1, :])
		his_vector = torch.stack(H_cas, dim=0)   # (batch_size, emb_dim)'''
		
		dy_his_vector = F.embedding(history, graph_user_embedding)
		dy_his_vector = dy_his_vector.sum(1) / lengths[:, None].float()
		his_vector, his_gate = self.his_fusion(dy_his_vector, his_vector)
		
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
		return {'prediction': prediction.view(batch_size, -1)}
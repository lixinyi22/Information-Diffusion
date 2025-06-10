# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from models.GraphEncoder import *
from models.TransformerBlock import TransformerBlock
from utils import layers

class DyHGCN(SequentialModel):
	extra_log_args = ['emb_size', 'num_layers', 'num_heads']
	reader = 'DyHGCNReader'

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
		self.time_attention = TimeAttention(8, self.emb_size)
		self.linear = nn.Linear(self.emb_size + 8, self.emb_size)
		self.decoder_attention = TransformerBlock(input_size=self.emb_size + 8, n_heads=8)
		self.gnn_diffusion_layer = DynamicGraphNN(self.item_num, self.emb_size)
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
		dynamic_node_emb_dict = self.gnn_diffusion_layer(diffusion_graph)
		memory_emb_list = self.gnn_diffusion_layer(diffusion_graph)

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
		dyuser_emb_list = list() 
		for val in sorted(dynamic_node_emb_dict.keys()):
			dyuser_emb_sub = F.embedding(i_ids, dynamic_node_emb_dict[val].cuda()).unsqueeze(2)
			dyuser_emb_list.append(dyuser_emb_sub)
		dyuser_emb = torch.cat(dyuser_emb_list, dim=2)
		dy_i_vectors = dyuser_emb.mean(dim=2)
		i_vectors, i_gate = self.item_fusion(dy_i_vectors, i_vectors)
  
		# DyHGCN-global_his
		dyuser_emb_list = list() 
		for val in sorted(dynamic_node_emb_dict.keys()):
			dyuser_emb_sub = F.embedding(history, dynamic_node_emb_dict[val].cuda()).unsqueeze(2)
			dyuser_emb_list.append(dyuser_emb_sub)
		dyuser_emb = torch.cat(dyuser_emb_list, dim=2)
		dy_his_vector = dyuser_emb.mean(dim=2)

		# DyHGCN-local
		'''dyemb_timestamp = torch.zeros(batch_size, seq_len).long()
		dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
		dynamic_node_emb_dict_time_dict = dict()
		for i, val in enumerate(dynamic_node_emb_dict_time):
			dynamic_node_emb_dict_time_dict[val] = i 
		latest_timestamp = dynamic_node_emb_dict_time[-1]
		step_len = 5
		for t in range(0, seq_len, step_len):
			try:
				la_timestamp = torch.max(timestamp[:, t:t+step_len]).item()
				if la_timestamp < 1:
					break 
				latest_timestamp = la_timestamp 
			except Exception:
				pass 

			res_index = len(dynamic_node_emb_dict_time_dict)-1
			for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
				if val <= latest_timestamp:
					res_index = i  
					continue 
				else:
					break
			dyemb_timestamp[:, t:t+step_len] = res_index
		batch_t = torch.arange(history.size(1)).expand(history.size()).cuda()
		order_embed = self.dropout_layer(self.pos_embedding(batch_t))
		dyemb = self.time_attention(dyemb_timestamp.cuda(), his_vectors.unsqueeze(2))
		dyemb = self.dropout_layer(dyemb) 
		final_embed = torch.cat([dyemb, order_embed], dim=-1).cuda() # dynamic_node_emb
		att_out = self.decoder_attention(final_embed.cuda(), final_embed.cuda(), final_embed.cuda(), mask=mask.cuda())
		his_vector = att_out[torch.arange(batch_size), lengths - 1, :]
		his_vector = self.linear(his_vector)'''
		
		dy_his_vector = dy_his_vector * valid_his[:, :, None].float()
		dy_his_vector = dy_his_vector.sum(1) / lengths[:, None].float()
		his_vector, his_gate = self.his_fusion(dy_his_vector, his_vector)
		
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
		return {'prediction': prediction.view(batch_size, -1)}
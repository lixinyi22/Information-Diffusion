# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import SequentialModel
from models.GraphEncoder import *
from models.TransformerBlock import TransformerBlock
from utils import layers

class MSHGAT(SequentialModel):
	extra_log_args = ['emb_size', 'num_layers', 'num_heads']
	reader = 'MSHGATReader'

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
		self.decoder_attention = TransformerBlock(input_size=self.emb_size + 8, n_heads=8)
		self.gnn_diffusion_layer = HGNN_ATT(input_size=64, output_size=self.emb_size, dropout=self.dropout)
		
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
		dy_i_vectors = F.embedding(i_ids, list(memory_emb_list.values())[-1][0].cuda())
		i_vectors, i_gate = self.item_fusion(dy_i_vectors, i_vectors)
		
		# MS-HGAT-global_his
		zero_vec = torch.zeros_like(history)
		dyemb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()

		for ind, time in enumerate(sorted(memory_emb_list.keys())):
			if ind == 0:
				sub_input = torch.where(timestamp <= time, history, zero_vec)
				sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())
				temp = sub_input == 0
			else:
				cur = torch.where(timestamp <= time, history, zero_vec) - sub_input
				temp = cur == 0
				sub_emb = F.embedding(cur.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())
				sub_input = cur + sub_input

			sub_emb[temp] = 0
			dyemb += sub_emb

			if ind == len(memory_emb_list) - 1:
				sub_input = history - sub_input
				temp = sub_input == 0
				sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind][0].cuda())
				sub_emb[temp] = 0

				dyemb += sub_emb
		# MS-HGAT-local
		'''batch_t = torch.arange(history.size(1)).expand(history.size()).cuda()
		order_embed = self.dropout_layer(self.pos_embedding(batch_t))
		diff_embed = torch.cat([his_vectors, order_embed], dim=-1).cuda()
		diff_att_out = self.decoder_attention(diff_embed.cuda(), diff_embed.cuda(), diff_embed.cuda(), mask=mask.cuda())
		his_vector = diff_att_out[torch.arange(batch_size), lengths - 1, :]
		his_vector = self.linear(his_vector)'''
		
		# his fusion
		dy_his_vector = dyemb * valid_his[:, :, None].float()
		dy_his_vector = dy_his_vector.sum(1) / lengths[:, None].float()
		his_vector, his_gate = self.his_fusion(dy_his_vector, his_vector)
		
		prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
		return {'prediction': prediction.view(batch_size, -1)}
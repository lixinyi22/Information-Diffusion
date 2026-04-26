import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from model.layers import TransformerBlock, IntervalTimeEncoder
import utils.Constants as Constants

class MyModel(nn.Module):
    def __init__(self, opt, content_embedding, user_info_participation=None):
        super(MyModel, self).__init__()
        self.hidden_size = opt.d_model
        self.n_node = opt.user_size
        self.dropout = nn.Dropout(opt.dropout)
        self.emb_size = opt.d_model
        self.max_his = opt.max_len
        self.num_seq_layers = opt.num_seq_layers
        self.num_heads = opt.num_heads
        self.knn_k = opt.knn_k
        self.num_mm_layers = opt.num_mm_layers
        self.num_hg_layers = opt.num_hg_layers
        self.t_feat = content_embedding
        self.ablation_mode = getattr(opt, "ablation_mode", "full")
        self.fusion_mode = getattr(opt, "fusion_mode", "sum")
        self.disable_self_gate = getattr(opt, "disable_self_gate", False)
        self.use_history_user_content = getattr(opt, "use_history_user_content", False)
        self.use_user_content_emb = self.fusion_mode in ("sum_uc", "concat_uc") or self.ablation_mode == "only_contextual"
        self.is_concat_fusion = self.fusion_mode in ("concat", "concat_uc")
        self.fusion_weight_mode = getattr(opt, "fusion_weight_mode", "disable")
        self.fusion_w_g = float(getattr(opt, "fusion_w_g", 1.0))
        self.fusion_w_i = float(getattr(opt, "fusion_w_i", 1.0))
        self.fusion_w_c = float(getattr(opt, "fusion_w_c", 1.0))
        if self.fusion_weight_mode == "adaptive":
            g, i, c = self.fusion_w_g, self.fusion_w_i, self.fusion_w_c
            init = torch.log(
                torch.tensor(
                    [max(g, 1e-8), max(i, 1e-8), max(c, 1e-8)],
                    dtype=torch.float32,
                )
            )
            self.fusion_weight_logits = nn.Parameter(init)
        else:
            self.fusion_weight_logits = None
        if user_info_participation is None:
            user_info_participation = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
                (self.n_node, opt.info_size),
            )
        if not user_info_participation.is_sparse:
            user_info_participation = user_info_participation.to_sparse()
        self.register_buffer("user_info_participation", user_info_participation.coalesce(), persistent=False)
        if self.ablation_mode == "only_structural":
            self.use_structural, self.use_interactive, self.use_contextual = True, False, False
        elif self.ablation_mode == "only_interactive":
            self.use_structural, self.use_interactive, self.use_contextual = False, True, False
        elif self.ablation_mode == "only_contextual":
            self.use_structural, self.use_interactive, self.use_contextual = False, False, True
        else:
            self.use_structural = self.ablation_mode != "wo_structural"
            self.use_interactive = self.ablation_mode != "wo_interactive"
            self.use_contextual = self.ablation_mode != "wo_contextual"


        # structural channel
        self.time_encoder = IntervalTimeEncoder(opt.time_interval, opt.time_step_split, opt.d_model)
        self.decoder_attention = TransformerBlock(input_size=(self.hidden_size + opt.d_model*2), n_heads=self.num_heads)
        self.linear = nn.Linear(self.hidden_size + opt.d_model*2, self.emb_size)
        #self.user_graph_embedding = nn.Embedding(self.n_node, self.emb_size)
        #self.order_embedding = nn.Embedding(self.max_his + 1, self.emb_size)

        # interactive channel
        self.len_range = torch.from_numpy(np.arange(self.max_his)).cuda()
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size)
        #self.user_inter_embedding = nn.Embedding(self.n_node, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        if self.disable_self_gate:
            # Ablation: remove the shared base embedding + self-gate coupling.
            # Two channels now own fully independent base embeddings.
            self.user_graph_embedding = nn.Embedding(self.n_node, self.emb_size)
            self.user_inter_embedding = nn.Embedding(self.n_node, self.emb_size)
            self.p_graph_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
            self.p_inter_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            TransformerBlock(input_size=self.hidden_size, n_heads=self.num_heads, attn_dropout=opt.dropout)
            for _ in range(self.num_seq_layers)
        ])

        # contextual channel
        self.masked_adj, self.mm_adj = None, None
        self.info_id_embedding = nn.Embedding(opt.info_size, self.emb_size)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).cuda()
        _, self.mm_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.info_predict_linear = nn.Linear(self.emb_size, self.n_node)

        # InfoNCE contrastive learning parameters
        #self.temperature = opt.contrastive_temperature
        #self.contrastive_weight = opt.contrastive_weight

        # self-gating parameters
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(4)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(4)])

        # Legacy unused layers (kept for state_dict compatibility with older checkpoints: content_attn_*).
        self.content_attn_fc = nn.Linear(self.emb_size, self.emb_size)
        self.content_attn_query = nn.Linear(self.emb_size, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if name == "fusion_weight_logits":
                continue
            param.data.uniform_(-stdv, stdv)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).cuda()
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        # torch.sparse.FloatTensor(...) is deprecated; use sparse_coo_tensor instead
        return torch.sparse_coo_tensor(
            indices,
            values,
            adj_size,
            dtype=torch.float32,
            device=values.device,
        )

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def _build_channel_embeddings(self):
        if self.disable_self_gate:
            return (
                self.user_graph_embedding.weight,
                self.user_inter_embedding.weight,
                self.p_graph_embeddings.weight,
                self.p_inter_embeddings.weight,
            )
        return (
            self.self_gating(self.user_embedding.weight, 0),
            self.self_gating(self.user_embedding.weight, 1),
            self.self_gating(self.p_embeddings.weight, 2),
            self.self_gating(self.p_embeddings.weight, 3),
        )

    def _aggregate_user_content_emb(self, info_item_emb):
        """Aggregate user content embeddings from history interacted info IDs."""
        user_info_mat = self.user_info_participation
        if user_info_mat.device != info_item_emb.device:
            user_info_mat = user_info_mat.to(info_item_emb.device)
        user_info_mat = user_info_mat.coalesce()
        if user_info_mat._nnz() == 0:
            return torch.zeros(
                user_info_mat.size(0),
                self.emb_size,
                device=info_item_emb.device,
                dtype=info_item_emb.dtype,
            )

        indices = user_info_mat.indices()
        user_ids = indices[0]
        info_ids = indices[1]

        info_key = torch.tanh(self.content_attn_fc(info_item_emb))
        edge_logits = self.content_attn_query(info_key[info_ids]).squeeze(-1)
        attn_sparse = torch.sparse_coo_tensor(
            indices,
            edge_logits,
            user_info_mat.size(),
            dtype=edge_logits.dtype,
            device=edge_logits.device,
        ).coalesce()
        attn_sparse = torch.sparse.softmax(attn_sparse, dim=1)
        edge_alpha = attn_sparse.values().unsqueeze(-1)

        user_content_emb = torch.zeros(
            user_info_mat.size(0),
            self.emb_size,
            device=info_item_emb.device,
            dtype=info_item_emb.dtype,
        )
        user_content_emb.index_add_(0, user_ids, info_item_emb[info_ids] * edge_alpha)
        return user_content_emb

    @staticmethod
    def _safe_l2_normalize(x, dim=-1):
        """Avoid NaN when a row is all zeros (ablation / unused channel)."""
        n = x.norm(p=2, dim=dim, keepdim=True).clamp(min=1e-12)
        return x / n

    def _get_fusion_weights(self, ref_tensor):
        """(wg, wi, wc, w_gi). disable: all 1; manual: fusion_w_*; adaptive: softmax(logits)*3 so wg+wi+wc=3."""
        device, dtype = ref_tensor.device, ref_tensor.dtype
        if self.fusion_weight_mode == "disable":
            one = torch.tensor(1.0, device=device, dtype=dtype)
            wg = wi = wc = one
        elif self.fusion_weight_mode == "adaptive":
            logits = self.fusion_weight_logits.to(device=device, dtype=dtype)
            w = F.softmax(logits, dim=0) * 3.0
            wg, wi, wc = w[0], w[1], w[2]
        else:
            wg = torch.tensor(self.fusion_w_g, device=device, dtype=dtype)
            wi = torch.tensor(self.fusion_w_i, device=device, dtype=dtype)
            wc = torch.tensor(self.fusion_w_c, device=device, dtype=dtype)
        w_gi = 0.5 * (wg + wi)
        return wg, wi, wc, w_gi

    def forward(self, input, input_timestamp, input_id, diffusion_graph):
        input = input[:, :-1]
        input_timestamp = input_timestamp[:, :-1]
        mask = (input == Constants.PAD)
        batch_size, seq_len = input.size()

        lengths = (~mask).sum(dim=1)  # [batch_size] - 每个样本中非PAD元素的数量
        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        valid_his = (input > 0).long()
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        user_graph_emb, user_inter_emb, pos_graph_emb, pos_inter_emb = self._build_channel_embeddings()
        pos_vectors = F.embedding(position, pos_inter_emb)
        order_embed = F.embedding(position, pos_graph_emb)
        #pos_vectors = F.embedding(position, self.p_embeddings.weight)
        #order_embed = F.embedding(position, self.order_embedding.weight)
        #user_graph_emb = self.user_graph_embedding.weight
        #user_inter_emb = self.user_inter_embedding.weight

        # structural channel
        if self.use_structural:
            user_graph_emb = self.structure_embed(diffusion_graph, user_graph_emb.cuda())
            info_graph_emb = F.embedding(input, user_graph_emb)
            time_embedding = self.time_encoder(input, input_timestamp)
            info_graph_timestamp_emb =  torch.cat([info_graph_emb, time_embedding, order_embed], dim=-1).cuda()
            info_graph_emb = self.decoder_attention(info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), mask=mask.cuda())
            info_graph_emb = self.dropout(self.linear(info_graph_emb))
        else:
            user_graph_emb = torch.zeros_like(user_graph_emb)
            info_graph_emb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()

        # interactive channel
        if self.use_interactive:
            his_vectors = F.embedding(input, user_inter_emb)
            his_vectors = his_vectors + pos_vectors
            for block in self.transformer_block:
                his_vectors = block(his_vectors, his_vectors, his_vectors, mask=mask.cuda())
            info_inter_emb = his_vectors
        else:
            user_inter_emb = torch.zeros_like(user_inter_emb)
            info_inter_emb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()
        #info_inter_emb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()

        # contextual channel
        if self.use_contextual:
            h = self.info_id_embedding.weight
            for i in range(self.num_mm_layers):
                h = torch.sparse.mm(self.mm_adj, h)
            info_content_emb = h[input_id]
        else:
            h = self.info_id_embedding.weight
            info_content_emb = torch.zeros(batch_size, self.emb_size).cuda()
        #info_content_emb = torch.zeros(batch_size, self.emb_size).cuda()

        user_content_emb = None
        if self.use_user_content_emb and self.use_history_user_content:
            user_content_emb = self._aggregate_user_content_emb(h)

        # prediction with parser-controlled fusion variants
        prediction = self.embedding_concat(
            user_graph_emb,
            user_inter_emb,
            info_graph_emb,
            info_inter_emb,
            info_content_emb,
            seq_len,
            user_content_emb=user_content_emb,
        )
        mask = self.get_previous_user_mask(input)
        prediction = prediction + mask
        return prediction.view(-1, prediction.size(-1)), 0

    def get_previous_user_mask(self, seq):
        ''' Mask previous activated users.'''
        user_size = self.n_node
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.cuda()
        masked_seq = previous_mask * seqs.data.float()

        # force the 0th dimension (PAD) to be masked
        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.cuda()
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.cuda()
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
        masked_seq = Variable(masked_seq, requires_grad=False)
        # print("masked_seq ",masked_seq.size())
        return masked_seq.cuda()

    def inference(self, input, input_timestamp, input_content_embedding, diffusion_graph):
        input = input[:, :-1]
        input_timestamp = input_timestamp[:, :-1]
        mask = (input == Constants.PAD)
        batch_size, seq_len = input.size()

        lengths = (~mask).sum(dim=1)  # [batch_size] - 每个样本中非PAD元素的数量
        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        valid_his = (input > 0).long()
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        user_graph_emb, user_inter_emb, pos_graph_emb, pos_inter_emb = self._build_channel_embeddings()
        pos_vectors = F.embedding(position, pos_inter_emb)
        order_embed = F.embedding(position, pos_graph_emb)
        #pos_vectors = F.embedding(position, self.p_embeddings.weight)
        #order_embed = F.embedding(position, self.order_embedding.weight)
        #user_graph_emb = self.user_graph_embedding.weight
        #user_inter_emb = self.user_inter_embedding.weight

        if self.use_structural:
            user_graph_emb = self.structure_embed(diffusion_graph, user_graph_emb.cuda(), is_training=False)
            info_graph_emb = F.embedding(input, user_graph_emb)
            time_embedding = self.time_encoder(input, input_timestamp)
            info_graph_timestamp_emb =  torch.cat([info_graph_emb, time_embedding, order_embed], dim=-1).cuda()
            info_graph_emb = self.decoder_attention(info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), mask=mask.cuda())
            info_graph_emb = self.dropout(self.linear(info_graph_emb))
        else:
            user_graph_emb = torch.zeros_like(user_graph_emb)
            info_graph_emb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()

        # interactive channel
        if self.use_interactive:
            his_vectors = F.embedding(input, user_inter_emb)
            his_vectors = his_vectors + pos_vectors
            for block in self.transformer_block:
                his_vectors = block(his_vectors, his_vectors, his_vectors, mask=mask.cuda())
            info_inter_emb = his_vectors
        else:
            user_inter_emb = torch.zeros_like(user_inter_emb)
            info_inter_emb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()
        #info_inter_emb = torch.zeros(batch_size, seq_len, self.emb_size).cuda()

        # contextual channel
        # 1. GCN on known items
        if self.use_contextual:
            h = self.info_id_embedding.weight
            for i in range(self.num_mm_layers):
                h = torch.sparse.mm(self.mm_adj, h)

            # 3. Info embedding for unseen items via kNN weighted aggregation
            unknown_content_norm = F.normalize(input_content_embedding, p=2, dim=-1)
            known_text_emb = F.normalize(self.text_embedding.weight, p=2, dim=-1)
            sim_scores = torch.mm(unknown_content_norm, known_text_emb.transpose(0, 1))
            _, topk_indices = torch.topk(sim_scores, self.knn_k, dim=-1)
            topk_sims = sim_scores.gather(1, topk_indices)
            topk_weights = F.softmax(topk_sims / 0.1, dim=-1)
            topk_embeddings = h[topk_indices]
            info_content_emb = torch.bmm(topk_weights.unsqueeze(1), topk_embeddings).squeeze(1)
        else:
            h = self.info_id_embedding.weight
            info_content_emb = torch.zeros(batch_size, self.emb_size).cuda()
        #info_content_emb = torch.zeros(batch_size, self.emb_size).cuda()

        user_content_emb = None
        if self.use_user_content_emb and self.use_history_user_content:
            user_content_emb = self._aggregate_user_content_emb(h)

        # prediction with parser-controlled fusion variants
        prediction = self.embedding_concat(
            user_graph_emb,
            user_inter_emb,
            info_graph_emb,
            info_inter_emb,
            info_content_emb,
            seq_len,
            user_content_emb=user_content_emb,
        )
        # normalize per-channel embeddings for embedding_dict (analysis / visualization use)
        valid_mask = ~mask
        info_content_emb_norm = self._safe_l2_normalize(info_content_emb, dim=-1)  # (batch, d)
        info_content_seq_norm = info_content_emb_norm.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, d)
        embedding_dict = {
            'user_graph_emb':   self._safe_l2_normalize(user_graph_emb,   dim=-1),
            'user_inter_emb':   self._safe_l2_normalize(user_inter_emb,   dim=-1),
            'info_graph_emb':   self._safe_l2_normalize(info_graph_emb,   dim=-1)[valid_mask],
            'info_inter_emb':   self._safe_l2_normalize(info_inter_emb,   dim=-1)[valid_mask],
            'info_content_emb': info_content_emb_norm,
            'info_content_seq': info_content_seq_norm[valid_mask],
        }

        mask = self.get_previous_user_mask(input)
        prediction = prediction + mask
        return prediction.view(-1, prediction.size(-1)), embedding_dict

    def embedding_concat(self, user_graph_emb, user_inter_emb, info_graph_emb, info_inter_emb, info_content_emb, seq_len, user_content_emb=None):
        # Single-channel proxies (paper IV): one prediction path each, no zero-tensor normalize issues.
        if self.ablation_mode == "only_contextual":
            info_c = self._safe_l2_normalize(info_content_emb, dim=-1)
            info_content_seq = info_c.unsqueeze(1).expand(-1, seq_len, -1)
            if self.use_history_user_content and user_content_emb is not None:
                user_content_emb = self._safe_l2_normalize(user_content_emb, dim=-1)
                return torch.matmul(info_content_seq, torch.transpose(user_content_emb, 1, 0))
            return self.info_predict_linear(info_content_seq)

        if self.ablation_mode == "only_structural":
            ug = self._safe_l2_normalize(user_graph_emb, dim=-1)
            ig = self._safe_l2_normalize(info_graph_emb, dim=-1)
            return torch.matmul(ig, torch.transpose(ug, 1, 0))

        if self.ablation_mode == "only_interactive":
            ui = self._safe_l2_normalize(user_inter_emb, dim=-1)
            ii = self._safe_l2_normalize(info_inter_emb, dim=-1)
            return torch.matmul(ii, torch.transpose(ui, 1, 0))

        user_graph_emb   = self._safe_l2_normalize(user_graph_emb,   dim=-1)
        user_inter_emb   = self._safe_l2_normalize(user_inter_emb,   dim=-1)
        info_graph_emb   = self._safe_l2_normalize(info_graph_emb,   dim=-1)
        info_inter_emb   = self._safe_l2_normalize(info_inter_emb,   dim=-1)

        info_content_emb = self._safe_l2_normalize(info_content_emb, dim=-1)
        info_content_seq = info_content_emb.unsqueeze(1).expand(-1, seq_len, -1)
        if self.use_user_content_emb:
            if not self.use_history_user_content or user_content_emb is None:
                # Legacy behavior: reuse classifier weights as user content embeddings.
                user_content_emb = self.info_predict_linear.weight
            user_content_emb = self._safe_l2_normalize(user_content_emb, dim=-1)
        else:
            user_content_emb = None

        wg, wi, wc, w_gi = self._get_fusion_weights(info_graph_emb)

        if self.is_concat_fusion:
            # Scale each channel block on the info side so each inner-product block is weighted linearly.
            info_fusion_emb = torch.cat(
                [wg * info_graph_emb, wi * info_inter_emb, wc * info_content_seq],
                dim=-1,
            )
            if user_content_emb is not None:
                user_fusion_emb = torch.cat(
                    [user_graph_emb, user_inter_emb, user_content_emb],
                    dim=-1,
                )
            else:
                user_fusion_emb = torch.cat(
                    [user_graph_emb, user_inter_emb, torch.ones_like(user_graph_emb)],
                    dim=-1,
                )
            prediction = torch.matmul(info_fusion_emb, torch.transpose(user_fusion_emb, 1, 0))
        else:
            # Default/legacy sum fusion keeps the original term expansion.
            vs_us = torch.matmul(info_graph_emb, torch.transpose(user_graph_emb, 1, 0))
            vi_ui = torch.matmul(info_inter_emb, torch.transpose(user_inter_emb, 1, 0))
            vs_ui = torch.matmul(info_graph_emb, torch.transpose(user_inter_emb, 1, 0))
            vi_us = torch.matmul(info_inter_emb, torch.transpose(user_graph_emb, 1, 0))
            vc_us = torch.matmul(info_content_seq, torch.transpose(user_graph_emb, 1, 0))
            vc_ui = torch.matmul(info_content_seq, torch.transpose(user_inter_emb, 1, 0))
            prediction = (
                wg * vs_us
                + wi * vi_ui
                + w_gi * (vs_ui + vi_us)
                + wc * (vc_us + vc_ui)
            )
            if user_content_emb is not None:
                vc_uc = torch.matmul(info_content_seq, torch.transpose(user_content_emb, 1, 0))
                prediction = prediction + wc * vc_uc

        return prediction

    def structure_embed(self, diffusion_graph, user_embed, is_training=True):
        diffusion_graph = diffusion_graph.cuda()

        if is_training:
            diffusion_graph = self._dropout_graph(diffusion_graph, keep_prob=0.9)
        else:
            diffusion_graph = diffusion_graph

        all_emb = [user_embed]

        for k in range(self.num_hg_layers):
            user_embed = torch.sparse.mm(diffusion_graph, user_embed)
            norm_embed = F.normalize(user_embed, p=2, dim=1)
            all_emb += [norm_embed]

        user_embed = torch.stack(all_emb, dim=1)
        user_embed = torch.sum(user_embed, dim=1)
        return user_embed

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()

        graph_coalesced = graph.coalesce()
        index = graph_coalesced.indices()  # (2, n_edges)，保持原始形状
        values = graph_coalesced.values()   # (n_edges,)

        random_mask = torch.rand(len(values), device=values.device) < keep_prob

        index = index[:, random_mask]  # (2, n_selected_edges)
        values = values[random_mask] / keep_prob

        g = torch.sparse_coo_tensor(index, values, size, dtype=torch.float32)
        return g

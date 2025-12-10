import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from model.layers import TransformerBlock, IntervalTimeEncoder
import utils.Constants as Constants

class MyModel(nn.Module):
    def __init__(self, opt, content_embedding):
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
        
        
        # structural channel
        self.time_encoder = IntervalTimeEncoder(opt.time_interval, opt.time_step_split, opt.d_model)
        self.decoder_attention = TransformerBlock(input_size=(self.hidden_size + opt.d_model*2), n_heads=self.num_heads)
        self.linear = nn.Linear(self.hidden_size + opt.d_model*2, self.emb_size)
        
        # interactive channel
        self.len_range = torch.from_numpy(np.arange(self.max_his)).cuda()
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

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
        self.temperature = opt.contrastive_temperature
        self.contrastive_weight = opt.contrastive_weight
        
        # self-gating parameters
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(4)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(4)])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

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
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def info_nce_loss(self, query, positive, negative=None):
        """
        InfoNCE对比学习损失函数
        
        Args:
            query: (batch_size, emb_size) 查询向量
            positive: (batch_size, emb_size) 正样本向量
            negative: (batch_size, num_negatives, emb_size) 负样本向量，如果为None则从batch中采样
            temperature: 温度参数，如果为None则使用self.temperature
        
        Returns:
            loss: 标量损失值
        """
        temperature = self.temperature
        
        # L2归一化：确保点积等于余弦相似度
        # 归一化后的向量，点积 = 余弦相似度（因为 ||v|| = 1）
        query = F.normalize(query, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        # 计算正样本相似度: (batch_size,)
        # 使用 torch.sum(query * positive, dim=-1) 而不是 torch.matmul 的原因：
        # 1. 只需要计算对角线元素（每个query与其对应的positive的点积）
        # 2. torch.sum(query * positive, dim=-1) 只计算 batch_size 个点积，更高效
        # 3. 等价于 torch.diag(torch.matmul(query, positive.t()))，但更节省内存
        pos_sim = torch.sum(query * positive, dim=-1)  # (batch_size,)
        pos_sim = pos_sim / temperature
        
        if negative is None:
            # 从batch中采样负样本：除了当前样本外的所有样本
            # 计算query与batch中所有样本的相似度
            # 使用 torch.matmul 计算所有query与所有positive的相似度矩阵
            # 对角线是正样本，其他是负样本
            all_sim = torch.matmul(query, positive.t()) / temperature  # (batch_size, batch_size)
            # 对角线是正样本，需要排除
            logits = all_sim
            labels = torch.arange(query.size(0), device=query.device)
        else:
            # 使用提供的负样本
            negative = F.normalize(negative, p=2, dim=-1)
            neg_sim = torch.bmm(query.unsqueeze(1), negative.transpose(1, 2)).squeeze(1) / temperature  # (batch_size, num_negatives)
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)
            labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def compute_contrastive_loss(self, info_graph_emb, info_inter_emb, info_content_emb, mask):
        """
        计算用户和信息embedding之间的对比学习损失
        使用info_nce_loss函数实现
        
        Args:
            info_graph_emb: (batch_size, seq_len, emb_size) 图结构信息embedding
            info_inter_emb: (batch_size, seq_len, emb_size) 交互序列信息embedding
            info_content_emb: (batch_size, seq_len, emb_size) 内容信息embedding
            input: (batch_size, seq_len) 输入序列
            mask: (batch_size, seq_len) PAD mask
        
        Returns:
            total_loss: 总对比损失
        """
        # 2. 信息embedding之间的对比学习
        # 对每个batch中的每个有效位置，计算三种info embedding之间的对比损失
        # 只考虑非PAD位置
        valid_mask = ~mask  # (batch_size, seq_len)
        # 提取有效位置的embedding
        info_graph_valid = info_graph_emb[valid_mask]  # (valid_positions, emb_size)
        info_inter_valid = info_inter_emb[valid_mask]  # (valid_positions, emb_size)
        info_content_valid = info_content_emb[valid_mask]  # (valid_positions, emb_size)
            
        # info_graph vs info_content
        loss_gc = self.info_nce_loss(info_graph_valid, info_content_valid)
        
        # info_inter vs info_content
        loss_ic = self.info_nce_loss(info_inter_valid, info_content_valid)
        
        info_contrastive_loss = loss_gc + loss_ic
        
        # 总对比损失
        total_contrastive_loss = info_contrastive_loss
        
        return total_contrastive_loss

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))


    def forward(self, input, input_timestamp, input_id, diffusion_graph):
        input = input[:, :-1]
        #input_timestamp = input_timestamp[:, :-1]
        mask = (input == Constants.PAD)
        batch_size, seq_len = input.size()
        
        lengths = (~mask).sum(dim=1)  # [batch_size] - 每个样本中非PAD元素的数量
        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        valid_his = (input > 0).long()
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_graph_emb = self.self_gating(self.p_embeddings.weight, 2)
        pos_inter_emb = self.self_gating(self.p_embeddings.weight, 3)
        pos_vectors = F.embedding(position, pos_inter_emb)
        order_embed = F.embedding(position, pos_graph_emb)
        
        user_graph_emb = self.self_gating(self.user_embedding.weight, 0)
        user_inter_emb = self.self_gating(self.user_embedding.weight, 1)
        
        # structural channel
        user_graph_emb = self.structure_embed(diffusion_graph, user_graph_emb.cuda())
        info_graph_emb = F.embedding(input, user_graph_emb)
        time_embedding = self.time_encoder(input, input_timestamp)
        info_graph_timestamp_emb =  torch.cat([info_graph_emb, time_embedding, order_embed], dim=-1).cuda()
        info_graph_emb = self.decoder_attention(info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), mask=mask.cuda())
        info_graph_emb = self.dropout(self.linear(info_graph_emb))
        
        # interactive channel
        his_vectors = F.embedding(input, user_inter_emb)
        his_vectors = his_vectors + pos_vectors
        for block in self.transformer_block:
            his_vectors = block(his_vectors, his_vectors, his_vectors, mask=mask.cuda())
        info_inter_emb = his_vectors
        
        # contextual channel
        h = self.info_id_embedding.weight
        for i in range(self.num_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        info_content_emb = h[input_id]
        
        # prediction
        # info: info_content_emb, info_inter_emb, info_graph_emb
        # user: user_graph_emb, user_inter_emb 
        user_graph_emb = F.normalize(user_graph_emb, p=2, dim=-1)  # (user_size, emb_size)
        user_inter_emb = F.normalize(user_inter_emb, p=2, dim=-1)  # (user_size, emb_size)
        info_graph_emb = F.normalize(info_graph_emb, p=2, dim=-1)  # (batch_size, seq_len, emb_size)
        info_inter_emb = F.normalize(info_inter_emb, p=2, dim=-1)  # (batch_size, seq_len, emb_size)
        info_content_emb = F.normalize(info_content_emb, p=2, dim=-1)  # (batch_size, emb_size)
        info_content_emb_expanded = info_content_emb.unsqueeze(1).expand(-1, seq_len, -1)
        prediction = self.embedding_concat(user_graph_emb, user_inter_emb, info_graph_emb, info_inter_emb, info_content_emb_expanded, seq_len)
        
        # 计算原有的对比学习损失
        contrastive_loss = self.compute_contrastive_loss(info_graph_emb, info_inter_emb, info_content_emb_expanded, mask)
        
        mask = self.get_previous_user_mask(input)
        prediction = prediction + mask
        return prediction.view(-1, prediction.size(-1)), contrastive_loss
    
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
        #input_timestamp = input_timestamp[:, :-1]
        mask = (input == Constants.PAD)
        batch_size, seq_len = input.size()
        
        lengths = (~mask).sum(dim=1)  # [batch_size] - 每个样本中非PAD元素的数量
        # Position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        valid_his = (input > 0).long()
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_graph_emb = self.self_gating(self.p_embeddings.weight, 2)
        pos_inter_emb = self.self_gating(self.p_embeddings.weight, 3)
        pos_vectors = F.embedding(position, pos_inter_emb)
        order_embed = F.embedding(position, pos_graph_emb)
            
        user_graph_emb = self.self_gating(self.user_embedding.weight, 0)
        user_inter_emb = self.self_gating(self.user_embedding.weight, 1)
        
        user_graph_emb = self.structure_embed(diffusion_graph, user_graph_emb.cuda(), is_training=False)
        info_graph_emb = F.embedding(input, user_graph_emb)
        time_embedding = self.time_encoder(input, input_timestamp)
        info_graph_timestamp_emb =  torch.cat([info_graph_emb, time_embedding, order_embed], dim=-1).cuda()
        info_graph_emb = self.decoder_attention(info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), info_graph_timestamp_emb.cuda(), mask=mask.cuda())
        info_graph_emb = self.dropout(self.linear(info_graph_emb))
        
        # interactive channel
        his_vectors = F.embedding(input, user_inter_emb)
        his_vectors = his_vectors + pos_vectors
        for block in self.transformer_block:
            his_vectors = block(his_vectors, his_vectors, his_vectors, mask=mask.cuda())
        info_inter_emb = his_vectors
        
        # contextual channel
        # 1. 先对已知的item进行图卷积
        h = self.info_id_embedding.weight
        for i in range(self.num_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        
        # 4. 对于未知的item，根据content_embedding生成embedding
        unknown_content_norm = F.normalize(input_content_embedding, p=2, dim=-1)
        known_text_emb = F.normalize(self.text_embedding.weight, p=2, dim=-1)
        sim_scores = torch.mm(unknown_content_norm, known_text_emb.transpose(0, 1))
        _, topk_indices = torch.topk(sim_scores, self.knn_k, dim=-1)
        topk_sims = sim_scores.gather(1, topk_indices)
        topk_weights = F.softmax(topk_sims / 0.1, dim=-1)
        topk_embeddings = h[topk_indices]
        info_content_emb = torch.bmm(topk_weights.unsqueeze(1), topk_embeddings).squeeze(1)
            
        # prediction
        # info: info_content_emb, info_inter_emb, info_graph_emb
        # user: user_graph_emb, user_inter_emb 
        user_graph_emb = F.normalize(user_graph_emb, p=2, dim=-1)  # (user_size, emb_size)
        user_inter_emb = F.normalize(user_inter_emb, p=2, dim=-1)  # (user_size, emb_size)
        info_graph_emb = F.normalize(info_graph_emb, p=2, dim=-1)  # (batch_size, seq_len, emb_size)
        info_inter_emb = F.normalize(info_inter_emb, p=2, dim=-1)  # (batch_size, seq_len, emb_size)
        info_content_emb = F.normalize(info_content_emb, p=2, dim=-1)  # (batch_size, emb_size)
        info_content_emb_expanded = info_content_emb.unsqueeze(1).expand(-1, seq_len, -1)
        prediction = self.embedding_concat(user_graph_emb, user_inter_emb, info_graph_emb, info_inter_emb, info_content_emb_expanded, seq_len)
        valid_mask = ~mask
        embedding_dict = {
            'user_graph_emb': user_graph_emb,
            'user_inter_emb': user_inter_emb,
            'info_graph_emb': info_graph_emb[valid_mask],
            'info_inter_emb': info_inter_emb[valid_mask],
            'info_content_emb': info_content_emb
        }
        
        mask = self.get_previous_user_mask(input)
        prediction = prediction + mask
        return prediction.view(-1, prediction.size(-1)), embedding_dict
    
    def embedding_concat(self, user_graph_emb, user_inter_emb, info_graph_emb, info_inter_emb, info_content_emb, seq_len):
        
        user_emb = user_inter_emb + user_graph_emb
        info_emb = info_content_emb + info_graph_emb + info_inter_emb
        prediction = torch.matmul(info_emb, torch.transpose(user_emb, 1, 0))
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
        
        g = torch.sparse.FloatTensor(index, values, size)
        return g
        
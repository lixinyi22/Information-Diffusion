import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as fn

class IntervalTimeEncoder(nn.Module):
    def __init__(self, time_interval, n_time_interval, emb_size):
        super(IntervalTimeEncoder, self).__init__()
        self.time_interval = time_interval
        self.n_time_interval = n_time_interval
        self.output_dim= emb_size
        self.linear_1= nn.Linear(self.n_time_interval+1, self.output_dim, bias=True).cuda()
        init.xavier_normal_(self.linear_1.weight)
        self.relu=nn.ReLU()


    def forward(self,inputs,timestamp):
        '''batch_size,max_len=inputs.size()

        pass_time=timestamp[:,1:]-timestamp[:,:-1]
        pass_time=fn.relu(((pass_time / self.time_interval) * self.n_time_interval).long())
        pass_time=pass_time.view(batch_size*max_len,1).cuda()

        time_embedding_one_hot=torch.zeros(batch_size*max_len, self.n_time_interval+1).cuda()
        time_embedding_one_hot=time_embedding_one_hot.scatter_(1, pass_time, 1).cuda()
        time_embedding = self.linear_1(time_embedding_one_hot)  # [batch_size, max_len, output_dim]
        time_embedding=time_embedding.view(batch_size, max_len, self.output_dim).cuda()
'''
        batch_size,max_len=inputs.size()
        device = timestamp.device  # 获取输入张量的设备

        # 计算时间间隔：timestamp[:,1:]-timestamp[:,:-1] 的形状是 (batch_size, max_len-1)
        # 第一个位置没有前一个时间戳，使用0或第一个时间戳本身
        if max_len > 1:
            # 计算相邻时间戳的差值
            time_diffs = timestamp[:,1:] - timestamp[:,:-1]  # (batch_size, max_len-1)
            # 第一个位置使用0（表示没有时间间隔），或者使用第一个时间戳
            # 这里使用0填充第一个位置
            first_time = torch.zeros(batch_size, 1, device=device, dtype=timestamp.dtype)
            pass_time = torch.cat([first_time, time_diffs], dim=1)  # (batch_size, max_len)
        else:
            # 如果 max_len <= 1，直接使用零
            pass_time = torch.zeros(batch_size, max_len, device=device, dtype=timestamp.dtype)

        # 将时间间隔转换为区间索引
        pass_time = fn.relu(((pass_time / self.time_interval) * self.n_time_interval).long())
        # 确保索引不超过 n_time_interval
        pass_time = torch.clamp(pass_time, 0, self.n_time_interval)
        pass_time = pass_time.view(batch_size*max_len, 1)

        time_embedding_one_hot = torch.zeros(batch_size*max_len, self.n_time_interval+1, device=device)
        time_embedding_one_hot = time_embedding_one_hot.scatter_(1, pass_time, 1)
        time_embedding = self.linear_1(time_embedding_one_hot)  # [batch_size*max_len, output_dim]
        time_embedding = time_embedding.view(batch_size, max_len, self.output_dim)

        return time_embedding

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=64, d_v=64, n_heads=2, is_layer_norm=True, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=input_size)

        # self.pos_encoding = PositionalEncoding(d_model=input_size, dropout=0.5)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        #print(self)

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(fn.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param mask: (*, max_q_words)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, K.size(1))
            mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().cuda()
            mask_ = mask + pad_mask
            Q_K = Q_K.masked_fill(mask_, -2**32+1)

        Q_K_score = fn.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score = self.dropout(Q_K_score)
        #维度为3的两个矩阵的乘法
        V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        return V_att


    def multi_head_attention(self, Q, K, V, mask):
        '''
        :param Q:
        :param K:
        :param V:
        :param mask: (bsz, max_q_words)
        :return:
        '''
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)  # For head axis broadcasting.
            mask = mask.reshape(-1, mask.size(-1))

        V_att = self.scaled_dot_product_attention(Q_, K_, V_, mask)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V, mask=None):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V, mask)

        if self.is_layer_norm:
            X = self.layer_norm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_norm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output
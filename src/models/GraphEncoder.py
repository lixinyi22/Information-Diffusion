import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
time_step_split = 8
n_heads = 14
step_len = 5 

class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)

        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)
        #init.xavier_normal_(self.gnn1.weight)
        #init.xavier_normal_(self.gnn2.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        # print (graph_x_embeddings.shape)
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        return graph_output.cuda()

class DynamicGraphNN(nn.Module):
    def __init__(self, ntoken, nhid, dropout=0.1):
        super(DynamicGraphNN, self).__init__()
        self.nhid = nhid
        self.ntoken = ntoken
        self.embedding = nn.Embedding(ntoken, nhid)
        init.xavier_normal_(self.embedding.weight)

        self.gnn1 = GraphNN(ntoken, nhid)
        self.linear = nn.Linear(nhid * time_step_split, nhid) 
        init.xavier_normal_(self.linear.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, diffusion_graph_list):
        res = dict()
        graph_embeddinng_list = list() 
        for key in sorted(diffusion_graph_list.keys()):
            graph = diffusion_graph_list[key] 
            graph_x_embeddings = self.gnn1(graph)
            graph_x_embeddings = self.drop(graph_x_embeddings)
            graph_x_embeddings = graph_x_embeddings.cpu()

            graph_embeddinng_list.append(graph_x_embeddings)
            res[key] = graph_x_embeddings
        return res

class Fusion(nn.Module):
    def __init__(self, emb_dim, dropout=0.2):
        super(Fusion, self).__init__()
        self.gate_layer = nn.Linear(2 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_static = nn.LayerNorm(emb_dim)
        self.norm_dynamic = nn.LayerNorm(emb_dim)
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.gate_layer.weight, nonlinearity='leaky_relu', a=0.1)
        init.xavier_normal_(self.gate_layer.weight)

    def forward(self, u_static, u_dynamic):
        # 拼接静态和动态嵌入
        # 对输入进行LayerNorm归一化（按embedding维度）
        u_static = self.norm_static(u_static)
        u_dynamic = self.norm_dynamic(u_dynamic)
        concat = torch.cat([u_static, u_dynamic], dim=-1) * 1.5 - 0.25  # 控制输出在[0.25, 1.25]
        
        # 生成门控值（每个维度独立权重）
        gate = torch.sigmoid(self.gate_layer(concat))  # [batch, emb_dim]
        
        # 加权融合
        u_fused = gate * u_static + (1 - gate) * u_dynamic
        return u_fused, gate

class SelfGating(nn.Module):
    def __init__(self, in_channels, out_channels = 1, dropout=0.0):
        super(SelfGating, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout
        self.linear = Lin(in_channels, in_channels)
        
    def forward(self, x, *args):
        return x * F.sigmoid(self.linear(x))
        
    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__, self.in_channels, self.in_channels)

class TimeAttention(nn.Module):
    def __init__(self, time_size, in_features1):
        super(TimeAttention, self).__init__()
        self.time_embedding = nn.Embedding(time_size, in_features1)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, T_idx, Dy_U_embed, mask=None, episilon=1e-6):
        '''
            T_idx: (bsz, user_len)
            Dy_U_embed: (bsz, user_len, time_len, d) # uid 从动态embedding lookup 之后的节点向量
            output: (bsz, user_len, d) 
        '''
        temperature = Dy_U_embed.size(-1) ** 0.5 + episilon
        T_embed = self.time_embedding(T_idx) # (bsz, user_len, d)

        # print(T_embed.size())
        # print(Dy_U_embed.size())

        affine = torch.einsum("bud,butd->but", T_embed, Dy_U_embed) # (bsz, user_len, time_len)
        score = affine / temperature 

        # if mask is None:
        #     mask = torch.triu(torch.ones(score.size()), diagonal=1).bool().cuda()
        #     score = score.masked_fill(mask, -2**32+1)

        alpha = F.softmax(score, dim=1)  # (bsz, user_len, time_len) 
        # alpha = self.dropout(alpha)
        alpha = alpha.unsqueeze(dim=-1)  # (bsz, user_len, time_len, 1) 

        att = (alpha * Dy_U_embed).sum(dim=2)  # (bsz, user_len, d) 
        return att 

class HGATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, concat=True, edge = True):
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.edge = edge

        self.weight3 = nn.Parameter(torch.Tensor(self.out_features, self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight3.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        '''
        if self.transfer:
            x = x.matmul(self.weight)
        else:
            x = x.matmul(self.weight2)
        
        if self.bias is not None:
            x = x + self.bias  '''
            
        #n2e_att = get_NodeAttention(x, adj.t(), root_emb)

        adjt = F.softmax(adj.T,dim = 1)
        #adj = normalize(adj)
        num_nodes = adj.size(0)
        x = torch.ones(num_nodes, self.out_features, device=adj.device)
        edge = torch.matmul(adjt, x)
        
        edge = F.dropout(edge, self.dropout, training=self.training)
        edge = F.relu(edge,inplace = False)

        e1 = edge.matmul(self.weight3)

        
        adj = F.softmax(adj,dim = 1)
        #adj = get_EdgeAttention(adj)

        node = torch.matmul(adj, e1)
        node = F.dropout(node, self.dropout, training=self.training)
        

        if self.concat:
            node = F.relu(node,inplace = False)
            
        if self.edge:
            edge = torch.matmul(adjt, node)        
            edge = F.dropout(edge, self.dropout, training=self.training)
            edge = F.relu(edge,inplace = False) 
            return node, edge
        else:
            return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.3, is_norm=True):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(output_size)
        self.gat1 = HGATLayer(input_size, output_size, dropout=self.dropout, concat=True, edge=True)
        #self.fus1 = Fusion(output_size)

    def forward(self, hypergraph_list):
        hypergraph_list = hypergraph_list[0]
        embedding_list = {}
        for sub_key in hypergraph_list.keys():
            sub_graph = hypergraph_list[sub_key]
            sub_node_embed, sub_edge_embed = self.gat1(sub_graph.cuda())
            sub_node_embed = F.dropout(sub_node_embed, self.dropout, training=self.training)

            if self.is_norm:
                sub_node_embed = self.batch_norm1(sub_node_embed)
                sub_edge_embed = self.batch_norm1(sub_edge_embed)

            #x = self.fus1(x, sub_node_embed)
            #embedding_list[sub_key] = [x.cpu(), sub_edge_embed.cpu()]
            embedding_list[sub_key] = [sub_node_embed.cpu(), sub_edge_embed.cpu()]
            
        return embedding_list

class HypergraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, hypergraph):
        X = self.theta(X)
        Y = hypergraph.v2e(X, aggr="mean")
        X_ = hypergraph.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_

class Fusion2(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion2, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        '''
        hidden: 这个子超图HGAT的输入，dy_emb: 这个子超图HGAT的输出
        hidden和dy_emb都是用户embedding矩阵，大小为(用户数, 64)
        '''
        # tensor.unsqueeze(dim) 扩展维度，返回一个新的向量，对输入的既定位置插入维度1
        # tensor.cat(inputs, dim=?) --> Tensor    inputs：待连接的张量序列     dim：选择的扩维，沿着此维连接张量序列
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = nn.functional.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)  # 随机丢弃每个用户embedding的权重
        out = torch.sum(emb_score * emb, dim=0)  # 将输入的embedding和输出的embedding按照对应的用户加权求和
        return out

class DynamicCasHGNN(nn.Module):
    '''超图HGNN'''
    def __init__(self, input_num, embed_dim, dropout=0.5):
        '''
        :param input_num: 用户个数
        :param embed_dim: embedding维度
        :param step_split: 超图序列中的超图个数
        :param dropout: 丢弃率
        :param is_norm: 是否规则化
        '''
        super().__init__()
        self.input_num = input_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.user_embeddings = nn.Embedding(self.input_num, self.embed_dim)
        self.hgnn = HypergraphConv(self.embed_dim, self.embed_dim, drop_rate=self.dropout)  # 超图卷积，学习每个超图中的用户embedding
        # self.lstm = nn.LSTM(self.embed_dim, self.embed_dim, num_layers=1, batch_first=True) # LSTM学习超图间的关系
        self.fus = Fusion2(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        '''从正态分布中随机初始化每张超图的初始用户embedding'''
        init.xavier_normal_(self.user_embeddings.weight)

    def forward(self, hypergraph_list):
        # 对每张子超图进行卷积
        hg_embeddings = []
        for i in range(len(hypergraph_list)):
            subhg_embedding = self.hgnn(self.user_embeddings.weight, hypergraph_list[i])
            if i == 0:
                hg_embeddings.append(subhg_embedding)
            else:
                subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
                hg_embeddings.append(subhg_embedding)

            # print(f'self.user_embeddings[{i}].weight = {self.user_embeddings[i].weight}')
        # 返回最后一个时刻的用户embedding
        return hg_embeddings[-1]

class CascadeLSTM(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, num_layers=1, batch_first=True)

    def lookup_embedding(self, examples, embeddings):
        output_embedding = []
        for example in examples:
            index = example.clone().detach()
            temp = torch.index_select(embeddings, dim=0, index=index)
            output_embedding.append(temp)
        output_embedding = torch.stack(output_embedding, 0)
        return output_embedding

    def forward(self, examples, user_cas_embedding):
        '''
        :param examples: tensor 级联序列 (batch_size, 200)
        :param user_cas_embedding: tensor 动态级联图中的用户embedding (user_size, emb_dim)
        :return:
        '''
        cas_embedding = self.lookup_embedding(examples, user_cas_embedding)
        # output.size()=(input_num, step_split, embed_dim)
        # h.size()=(1, input_num, embed_dim) lstm中的参数
        # c.size()=(1, input_num, embed_dim) lstm中的参数
        output_embedding, (h_t, c_t) = self.lstm(cas_embedding)

        return output_embedding

class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1

        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class LSTMGNN(nn.Module):
    def __init__(self, emb_size, item_num, dropout=0.2):
        super(LSTMGNN, self).__init__()

        # parameters
        self.emb_size = emb_size
        self.n_node = item_num
        self.layers = 3
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = 2 
        self.win_size = 5

        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)

        ### channel self-gating parameters
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        self.reset_parameters()

        #### optimizer and loss function
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, 1000)
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, hypergraphs, phase, device):
        #hypergraph
        self.H_Item = hypergraphs[0].to(device)   
        self.H_User = hypergraphs[1].to(device)
        
        if phase == 'train':
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1-self.drop_rate)
            H_User = self._dropout_graph(self.H_User, keep_prob=1-self.drop_rate)
        else:
            H_Item = self.H_Item
            H_User = self.H_User

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)

        all_emb_c2 = [u_emb_c2]
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            # Channel Item
            u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)
            norm_embeddings2 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings2]

        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)
        u_emb_c3 = torch.stack(all_emb_c3, dim=1)
        u_emb_c3 = torch.sum(u_emb_c3, dim=1)

        # aggregating channel-specific embeddings
        high_embs, attention_score = self.channel_attention(u_emb_c2, u_emb_c3)

        return high_embs

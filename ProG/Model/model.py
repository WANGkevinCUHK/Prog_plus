import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv,GINConv,SAGEConv
from torch_geometric.nn import GraphConv as GConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.cluster import KMeans

from ..utils import act
from torch_geometric.nn import MessagePassing

    
class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type='mean'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        
        # 输入层
        if n_layers > 1:
            self.convs.append(SAGEConv(in_feats, n_hidden, normalize=False, aggr=aggregator_type))
        else:
            self.convs.append(SAGEConv(in_feats, n_classes, normalize=False, aggr=aggregator_type))
            
        # 隐藏层
        for _ in range(1, n_layers - 1):
            self.convs.append(SAGEConv(n_hidden, n_hidden, normalize=False, aggr=aggregator_type))
        
        # 输出层
        if n_layers > 1:
            self.convs.append(SAGEConv(n_hidden, n_classes, normalize=False, aggr=aggregator_type))
        
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
# ###
# forward函数依次实现了从第L层到第1层采样得到的bipartite子图的卷积。
# adjs包含L层邻居采样的bipartite子图：(edge_index, e_id, size), SAGEConv是支持bipartite图的。
# 对bipartite图进行卷积时，输入的x是一个tuple: (x_source, x_target)。
# 上面提到过，n_id是包含所有在L层卷积中遇到的节点的list，且target节点在n_id前几位。而bipartite图的size是(num_of_source_nodes, num_of_target_nodes)，
# 因此对每一层的bipartite图都有x_target = x[:size[1]] 。所以 self.convs[i]((x, x_target), edge_index)实现了对一层bipartite图的卷积。
# ###
    def forward(self, x, adjs):
        x = self.dropout(x)
        for i, (layer,adj) in enumerate(zip(self.layers,adjs)):
            x = layer(x, adj.edge_index)
            if i != self.n_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return x
    
    def predict(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # 创建正样本标签（1）和负样本标签（0）
        pos_neg_labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))],dim=0)
        z_i = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return self.prediction(z_i), pos_neg_labels

    
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3,JK="last", drop_ratio=0, pool='mean', gnn_type='GAT'):
        super().__init__()
        """
        Args:
            num_layer (int): the number of GNN layers
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            pool (str): sum, mean, max, attention, set2set
            
        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
        """
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        elif gnn_type == 'GraphSage':
            GraphConv = SAGEConv
        elif gnn_type == 'GConv':
            GraphConv = GConv
        # The graph neural network operator from the "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks" paper.
        elif gnn_type == 'GIN':
            GraphConv = GINConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN, GraphSage, GConv and TransformerConv')

        
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.gnn_type = gnn_type

    def forward(self, x, edge_index, batch = None, prompt = None):
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            if idx == 0 and prompt is not None:
                x = prompt.add(x)
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]
        
        if batch == None:
            return node_emb
        else:
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GPPT(nn.Module):
    def __init__(self, in_feats, n_hidden=128, n_classes=None, n_layers=2, activation = F.relu, dropout=0.5, center_num=3):
        super(GPPT, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes=n_classes
        self.center_num=center_num
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden))

        self.prompt=nn.Linear(n_hidden,self.center_num,bias=False)
        
        self.pp = nn.ModuleList()
        for i in range(self.center_num):
            self.pp.append(nn.Linear(2*n_hidden,n_classes,bias=False))
        
        
    def model_to_array(self,args):
        s_dict = torch.load('./data_smc/'+args.dataset+'_model_'+args.file_id+'.pt')#,map_location='cuda:0')
        keys = list(s_dict.keys())
        res = s_dict[keys[0]].view(-1)
        for i in np.arange(1, len(keys), 1):
            res = torch.cat((res, s_dict[keys[i]].view(-1)))
        return res
    def array_to_model(self, args):
        arr=self.model_to_array(args)
        m_m=torch.load('./data_smc/'+args.dataset+'_model_'+args.file_id+'.pt')#,map_location='cuda:0')#+str(args.gpu))
        indice = 0
        s_dict = self.state_dict()
        for name, param in m_m.items():
            length = torch.prod(torch.tensor(param.shape))
            s_dict[name] = arr[indice:indice + length].view(param.shape)
            indice = indice + length
        self.load_state_dict(s_dict)
    
    def load_parameters(self, args):
        self.args=args
        self.array_to_model(args)

    def weigth_init(self, x, edge_index, label,index):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.activation(h)
        
        features=h[index]
        labels=label[index.long()]
        cluster = KMeans(n_clusters=self.center_num, n_init=10, random_state=0).fit(features.detach().cpu())
        
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data = temp.clone().detach()
        

        p=[]
        for i in range(self.n_classes):
            p.append(features[labels==i].mean(dim=0).view(1,-1))
        temp=torch.cat(p,dim=0)
        for i in range(self.center_num):
            self.pp[i].weight.data = temp.clone().detach()
        
    
    def update_prompt_weight(self,h):
        cluster = KMeans(n_clusters=self.center_num, n_init=10, random_state=0).fit(h.detach().cpu())
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data = temp.clone().detach()

    def get_mul_prompt(self):
        pros=[]
        for name,param in self.named_parameters():
            if name.startswith('pp.'):
                pros.append(param)
        return pros
        
    def get_prompt(self):
        for name,param in self.named_parameters():
            if name.startswith('prompt.weight'):
                pro=param
        return pro
    
    def get_mid_h(self):
        return self.fea

    def forward(self, x, edge_index):

        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if l != len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        x = self.activation(x)
        
        self.fea=x 
        h = x
        out=self.prompt(h)
        index=torch.argmax(out, dim=1)
        out=torch.FloatTensor(h.shape[0],self.n_classes).cuda()
        for i in range(self.center_num):
            out[index==i]=self.pp[i](h[index==i])
        return out
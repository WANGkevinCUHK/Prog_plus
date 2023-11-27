import torch as th
import torch.nn as nn
import torch.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv,GINConv,GraphSAGEConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.cluster import KMeans
import os
import sys
from prompt import GPPTPrompt
from utils import act
from torch_geometric.nn import MessagePassing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    
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

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,center_num):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes=n_classes
        self.center_num=center_num
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))

        # self.prompt=nn.Linear(n_hidden,self.center_num,bias=False)
        
        # self.pp = nn.ModuleList()
        # for i in range(self.center_num):
        #     self.pp.append(nn.Linear(2*n_hidden,n_classes,bias=False))
        self.prompt_module = GPPTPrompt(n_hidden, center_num, n_classes)
        #replace with GPPTPrompt in prompt.py
        
        
    def model_to_array(self,args):
        s_dict = torch.load('.pre_trained_gnn/'+args.dataset+'_model_'+args.file_id+'GCN.pt')#,map_location='cuda:0')
        keys = list(s_dict.keys())
        res = s_dict[keys[0]].view(-1)
        for i in np.arange(1, len(keys), 1):
            res = torch.cat((res, s_dict[keys[i]].view(-1)))
        return res
    def array_to_model(self, args):
        arr=self.model_to_array(args)
        m_m=torch.load('.pre_trained_gnn/'+args.dataset+'_model_'+args.file_id+'GCN.pt')#,map_location='cuda:0')#+str(args.gpu))
        indice = 0
        s_dict = self.state_dict()
        for name, param in m_m.items():
            length = torch.prod(torch.tensor(param.shape))
            s_dict[name] = arr[indice:indice + length].view(param.shape)
            indice = indice + length
        s_dict = {k: v for k, v in s_dict.items() if 'fc' not in k}
        self.load_state_dict(s_dict, strict=False)
    
    def load_parameters(self, args):
        self.args=args
        self.array_to_model(args)

    def weigth_init(self,data,feature,label,index):
        self.prompt_module.weigth_init(self,data,feature,label,index)
        
    
    def update_prompt_weight(self, h):
        self.prompt_module.update_prompt_weight(h)

    def get_mul_prompt(self):
        return self.prompt_module.get_mul_prompt()

    def get_prompt(self):
        return self.prompt_module.get_prompt()
    
    
    def get_mid_h(self):
        return self.fea

    def forward(self, adjs, inputs):
        if self.dropout==False:
            h=inputs
        else:
            h = self.dropout(inputs)
        for l, (adj,layer) in enumerate(zip(adjs,self.layers)):
            h_dst = h[:adj.size[1]]
            h = layer((h, h_dst),adj.edge_index)

            if l != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout!=False:
                    h = self.dropout(h)

        h = self.activation(h)
        # h_dst = self.activation(h_dst)
        # neighbor=h_dst
        # h=torch.cat((h,neighbor),dim=1)
        self.fea=h 
        out=self.prompt(h)
        index=torch.argmax(out, dim=1)
        out=torch.FloatTensor(h.shape[0],self.n_classes).cuda()
        for i in range(self.center_num):
            out[index==i]=self.pp[i](h[index==i])
        return out
    
class GraphSAGEConv(MessagePassing):

    def __init__(self, emb_dim, aggr="mean", input_layer=False):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        ### Mapping 0/1 edge features to embedding
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:, 7] = 1  # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_encoder(edge_attr)

        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        elif gnn_type == 'GIN':
            GraphConv == GINConv
        elif gnn_type == 'GraphSage':
            GraphConv == GraphSAGEConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb

class GPF_GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add", input_layer=input_layer))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, input_layer=input_layer))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, input_layer=input_layer))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, input_layer=input_layer))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, x, edge_index, edge_attr, prompt=None):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr, prompt)
            if layer == self.num_layer - 1:
                # remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        return node_representation
    
class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin",
                 final_linear=True, layer_number=1):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.final_linear = final_linear

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GPF_GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        if final_linear:
            self.graph_pred_linear = torch.nn.ModuleList()

            for i in range(layer_number - 1):
                self.graph_pred_linear.append(torch.nn.Linear(2 * self.emb_dim, 2 * self.emb_dim))
                self.graph_pred_linear[-1].reset_parameters()
                self.graph_pred_linear.append(torch.nn.ReLU())

            self.graph_pred_linear.append(torch.nn.Linear(2 * self.emb_dim, self.num_tasks))
            self.graph_pred_linear[-1].reset_parameters()

            # self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.gnn.load_state_dict(torch.load(model_file, map_location='cpu'))

    def forward(self, data, prompt=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_representation = self.gnn(x, edge_index, edge_attr, prompt)

        pooled = self.pool(node_representation, batch)
        center_node_rep = node_representation[data.center_node_idx]

        graph_rep = torch.cat([pooled, center_node_rep], dim=1)

        emb = graph_rep

        if self.final_linear:
            for i in range(len(self.graph_pred_linear)):
                emb = self.graph_pred_linear[i](emb)

        # return self.graph_pred_linear(graph_rep)
        return emb

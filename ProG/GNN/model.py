import torch as th
import torch.nn as nn
import torch.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.cluster import KMeans
import os
import sys
from prompt import GPPTPrompt
from utils import act
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
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
    
def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test

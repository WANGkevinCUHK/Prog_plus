import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from utils import act
import warnings
from deprecated.sphinx import deprecated
from sklearn.cluster import KMeans
from GNN.model import GNN
from torch_geometric.nn.inits import glorot

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            # pg_x = pg.x.to(device)
            # g_x = g.x.to(device)
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch

class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre

@deprecated(version='1.0', reason="Pipeline is deprecated, use FrontAndHead instead")
class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'):
        warnings.warn("deprecated", DeprecationWarning)

        super().__init__()
        # load pre-trained GNN
        self.gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        self.gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in self.gnn.parameters():
            p.requires_grad = False

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch: Batch):
        prompted_graph = self.PG(graph_batch)
        graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre

class GPPTPrompt(torch.nn.Module):
    def __init__(self, n_hidden, center_num, n_classes):
        super(GPPTPrompt, self).__init__()
        self.center_num = center_num
        self.prompt = torch.nn.Linear(n_hidden, center_num, bias=False)
        self.pp = torch.nn.ModuleList()
        for i in range(center_num):
            self.pp.append(torch.nn.Linear(2 * n_hidden, n_classes, bias=False))

    def weigth_init(self,data,feature,label,index):
        h = self.dropout(feature)
        edge_index = data.edge_index
        for l, layer in enumerate(self.layers):
            h = layer(h,edge_index)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.activation(h)
        data.h=h
        
        features=h[index]
        labels=label[index.long()]
        cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(features.detach().cpu())
        
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data = temp.clone().detach()


        p=[]
        for i in range(self.n_classes):
            p.append(features[labels==i].mean(dim=0).view(1,-1))
        temp=torch.cat(p,dim=0)
        for i in range(self.center_num):
            self.pp[i].weight.data = temp.clone().detach()    

    def update_prompt_weight(self,h):
        cluster = KMeans(n_clusters=self.center_num,random_state=0).fit(h.detach().cpu())
        temp=torch.FloatTensor(cluster.cluster_centers_).cuda()
        self.prompt.weight.data= temp.clone().detach()

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
        
class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb


class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p

class GPrompt(torch.nn.modules):
    def __init__(self):
        super(GPrompt,self).__init__()

    @staticmethod
    def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
        bsz = graph_sizes.size(0)
        dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

        min_size, max_size = graph_sizes.min(), graph_sizes.max()
        mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

        if min_size == max_size:
            return batched_graph_feats.view(bsz, max_size, -1), mask
        else:
            graph_sizes_list = graph_sizes.view(-1).tolist()
            unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
            for i, l in enumerate(graph_sizes_list):
                if l == max_size:
                    continue
                elif l > max_size:
                    unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
                else:
                    mask[i, l:].fill_(0)
                    zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                    unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
            return torch.stack(unbatched_graph_feats, dim=0), mask

    def Forward():
        pass
class Gprompt_layer_mean(GPrompt):
    def __init__(self):
        super(Gprompt_layer_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result

class Gprompt_layer_linear_mean(GPrompt):
    def __init__(self,input_dim,output_dim):
        super(Gprompt_layer_linear_mean, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.mean(dim=1)
        graph_prompt_result=torch.nn.functional.normalize(graph_prompt_result,dim=1)
        return graph_prompt_result

class Gprompt_layer_linear_sum(GPrompt):
    def __init__(self,input_dim,output_dim):
        super(Gprompt_layer_linear_sum, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.sum(dim=1)
        graph_prompt_result=torch.nn.functional.normalize(graph_prompt_result,dim=1)
        return graph_prompt_result



#sum result is same as mean result
class Gprompt_layer_sum(GPrompt):
    def __init__(self):
        super(Gprompt_layer_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(2, 2))
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result



class Gprompt_layer_weighted(GPrompt):
    def __init__(self,max_n_num):
        super(Gprompt_layer_weighted, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class Gprompt_layer_feature_weighted_mean(GPrompt):
    def __init__(self,input_dim):
        super(Gprompt_layer_feature_weighted_mean, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_embedding=graph_embedding*self.weight
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result

class Gprompt_layer_feature_weighted_sum(GPrompt):
    def __init__(self,input_dim):
        super(Gprompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        graph_embedding=graph_embedding*self.weight
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class Gprompt_layer_weighted_matrix(GPrompt):
    def __init__(self,max_n_num,input_dim):
        super(Gprompt_layer_weighted_matrix, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(input_dim,max_n_num))
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        #prompt: mean
        graph_prompt_result=graph_embedding.sum(dim=1)
        return graph_prompt_result

class Gprompt_layer_weighted_linear(GPrompt):
    def __init__(self,max_n_num,input_dim,output_dim):
        super(Gprompt_layer_weighted_linear, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,max_n_num))
        self.linear=torch.nn.Linear(input_dim,output_dim)
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight[0][0:graph_embedding.size(1)]
        temp1 = torch.ones(graph_embedding.size(0), graph_embedding.size(2), graph_embedding.size(1)).to(graph_embedding.device)
        temp1 = weight * temp1
        temp1 = temp1.permute(0, 2, 1)
        graph_embedding=graph_embedding*temp1
        graph_prompt_result = graph_embedding.mean(dim=1)
        return graph_prompt_result

class Gprompt_layer_weighted_matrix_linear(GPrompt):
    def __init__(self,max_n_num,input_dim,output_dim):
        super(Gprompt_layer_weighted_matrix_linear, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(output_dim,max_n_num))
        self.linear=torch.nn.Linear(input_dim,output_dim)
        self.max_n_num=max_n_num
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding, graph_len):
        graph_embedding=self.linear(graph_embedding)
        graph_embedding=self.split_and_batchify_graph_feats(graph_embedding, graph_len)[0]
        weight = self.weight.permute(1, 0)[0:graph_embedding.size(1)]
        weight = weight.expand(graph_embedding.size(0), weight.size(0), weight.size(1))
        graph_embedding = graph_embedding * weight
        graph_prompt_result=graph_embedding.mean(dim=1)
        return graph_prompt_result


if __name__ == '__main__':
    pass

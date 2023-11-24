import argparse
import random
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score
from get_args import get_my_args
import argparse
import random
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score
from get_args import get_my_args
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler

def GPPT_load_data(dataset):
    if dataset in ['Cora', 'CiteSeer']:
        dataset = Planetoid(root='/tmp/'+dataset, name=dataset)
        data = dataset[0]

        features = data.x
        labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        in_feats = features.size(1)
        n_classes = len(torch.unique(labels))
        n_edges = data.edge_index.size(1) // 2

        return data, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges

def evaluate(model, data, nid, batch_size, device,sample_list):
    valid_loader = NeighborSampler(data.edge_index, node_idx=nid, sizes=sample_list, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=0)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad(): 
        for step, (batch_size, n_id, adjs) in enumerate(valid_loader):      
            adjs = [adj.to(device) for adj in adjs]
            # 获取节点特征
            batch_features = data.x[n_id].to(device)
            # 获取节点标签（对于目标节点）
            batch_labels = data.y[n_id[:batch_size]].to(device)
            temp = model(adjs, batch_features).argmax(1)

            labels.append(batch_labels.cpu().numpy())
            predictions.append(temp.cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
    return accuracy
    
def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))
            
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_init_info(args,dataset,device):

    data,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges=my_load_data(dataset)
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))
    
        
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    # g = dgl.remove_self_loop(g)
    # n_edges = g.number_of_edges()

    return data,features,labels,in_feats,n_classes,train_nid,val_nid,test_nid,device
if __name__ == '__main__':
    args = get_my_args()
    print(args)
    info=my_load_data(args)
    g=info[0]
    labels=torch.sparse.sum(g.adj(),1).to_dense().int().view(-1,)
    print(labels)
    li=list(set(labels.numpy()))
    for i in range(labels.shape[0]):
        labels[i]=li.index(labels[i])
    print(set(labels.numpy()))
    #print(info)
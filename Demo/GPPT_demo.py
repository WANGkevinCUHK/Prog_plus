from GPPT.get_args import get_my_args
from ProG.utils import seed_torch,get_init_info,evaluate,get_init_info,constraint
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from GPPT.model import GraphSAGE
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.loader import NeighborSampler
import warnings
warnings.filterwarnings("ignore")

def main(args):
    seed_torch(args.seed)
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
    dataset = 'CiteSeer'

    data,features,labels,in_feats,n_classes,train_nid,val_nid,test_nid,device=get_init_info(args,dataset,device)

    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=args.sample_list, batch_size=args.batch_size, shuffle=True,drop_last=False,num_workers=0)
    model = GraphSAGE(in_feats,args.n_hidden,n_classes,args.n_layers,F.relu,args.dropout,args.aggregator_type,args.center_num)
    model.to(device)
    data.to(device)
    model.load_parameters(args)
    model.weigth_init(data,features,labels,train_nid)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_all=[]
    loss_all=[]
    for epoch in range(args.n_epochs):
        model.train()
        acc = evaluate(model, data, test_nid, args.batch_size, device, args.sample_list)
        acc_all.append(acc)
        t0 = time.time()
        for step, (batch_size, n_id, adjs) in enumerate(train_loader):
            optimizer.zero_grad()
            adjs = [adj.to(device) for adj in adjs]
            x = data.x[n_id].to(device)
            y = data.y[n_id].to(device)
            logits = model(adjs,x)
            lab = y[:batch_size]
            loss = F.cross_entropy(logits, lab)
            
            loss_all.append(loss.cpu().data)
            loss=loss+args.lr_c*constraint(device,model.get_mul_prompt())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            embedding_save=model.get_mid_h().detach().clone().cpu().numpy()
            Data=pd.DataFrame(embedding_save)
            label=pd.DataFrame(lab.detach().clone().cpu().numpy())
            Data.to_csv("./data.csv",index=None,header=None)
            label.to_csv("./label.csv",index=None,header=None)
            pd.DataFrame(torch.cat(model.get_mul_prompt(),axis=1).detach().clone().cpu().numpy()).to_csv("./data_p.csv",index=None,header=None)
            model.update_prompt_weight(model.get_mid_h())
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, time.time() - t0, loss.item(),acc))
    
    # pd.DataFrame(acc_all).to_csv('./res/gs_pre_pro_mul_pro_center_c_nei_'+args.dataset+'.csv',index=None,header=None)
    # pd.DataFrame(loss_all).to_csv('./res/gs_pre_pro_mul_pro_center_c_nei_'+args.dataset+'_loss.csv',index=None,header=None)


    acc = evaluate(model, data, test_nid, args.batch_size, device, args.sample_list)
    
    print("Test Accuracy {:.4f}".format(np.mean(acc_all[-10:])))


if __name__ == '__main__':
    args=get_my_args()
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

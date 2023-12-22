import argparse

from torch_geometric.data import InMemoryDataset
from ProG.Data.dataloader import DataLoaderAE
from ProG.utils import NegativeEdge
from ProG.get_args import get_pre_trained_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch_geometric.datasets import TUDataset
from ProG.Model.model import GNN
import pandas as pd

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_emb = model(batch.x, batch.edge_index, batch.edge_attr)

        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] * node_emb[batch.edge_index[1, ::2]], dim = 1)
        negative_score = torch.sum(node_emb[batch.negative_edge_index[0]] * node_emb[batch.negative_edge_index[1]], dim = 1)

        optimizer.zero_grad()
        loss = criterion(positive_score, torch.ones_like(positive_score)) + criterion(negative_score, torch.zeros_like(negative_score))
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step + 1)


def main():
    args = get_pre_trained_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    #set up dataset
    
    dataset = InMemoryDataset('data/TUDataset', transform = NegativeEdge())

    print(dataset)

    loader = DataLoaderAE(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    #optimizer = optim.Adam(model.graph_pred_linear.parameters(), lr=args.lr, weight_decay=args.decay)    
    print(optimizer)


    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, loader, optimizer)

        print(train_acc)
        print(train_loss)

    if not args.model_file == "":
        torch.save(model.state_dict(), args.model_file + ".pth")



if __name__ == "__main__":
    main()
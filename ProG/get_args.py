import argparse
from dgl.data import register_data_args
def get_my_args():
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,help="dropout probability")
    parser.add_argument("--lr_c", type=float, default=0.01,help="learning rate")
    parser.add_argument("--seed", type=int, default=100,help="random seed")
    parser.add_argument("--n-hidden", type=int, default=128,help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sample-list', type=list, default=[4,4])
    parser.add_argument("--n-epochs", type=int, default=30,help="number of training epochs")
    parser.add_argument("--file-id", type=str, default='128')
    parser.add_argument("--gpu", type=int, default=1,help="gpu")
    parser.add_argument("--lr", type=float, default=2e-3,help="learning rate")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--half', type=bool, default=False)
    parser.add_argument('--mask_rate', type=float, default=0)
    parser.add_argument('--center_num', type=int, default=7)
    args = parser.parse_args(args=['--dataset','citeseer'])
    print(args)
    return args

def get_pre_trained_args():
        # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 12, help='number of workers for dataset loading')
    args = parser.parse_args()
    return args

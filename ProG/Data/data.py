import torch
import pickle as pk
from random import shuffle

from torch_geometric.datasets import Planetoid, amazon, Reddit, WikiCS
from torch_geometric.data import Batch
from collections import defaultdict
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data

def multi_class_NIG(dataname, num_class,shots=100):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list,test_list

def load_graph_task(dataset_name, num_parts=None):
    if dataset_name == 'MUTAG' or 'ENZYMES' or 'COLLAB' or 'PROTEINS' or 'IMDB-BINARY'or 'REDDIT-BINARY':
        dataset = TUDataset(root='data/TUDataset', name=(dataset_name))

        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        data = dataset[0]  # Get the first graph object.

        print()
        print(data)
        print('=============================================================')

        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')


        torch.manual_seed(12345)
        dataset = dataset.shuffle()

        train_dataset = dataset[:100]
        test_dataset = dataset[100:]

        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')
        
        return dataset, train_dataset, test_dataset 

    if dataset_name == 'Cora' or 'Citesser':
        dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        num_parts=200

        x = data.x.detach()
        edge_index = data.edge_index
        edge_index = to_undirected(edge_index)
        data = Data(x=x, edge_index=edge_index)
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        
        graph_list = list(ClusterData(data=data, num_parts=num_parts))
        train_dataset = graph_list[:100]
        test_dataset = graph_list[100:]

        return input_dim, out_dim, train_dataset, test_dataset, graph_list 
    
def load_node_task(dataname):
    if dataname == 'PubMed' or 'CiteSeer' or 'Cora':
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
    if dataname == 'Computers' or 'Photo':
        dataset = amazon(root='data/amazon', name=dataname, transform=NormalizeFeatures())
    if dataname == 'Reddit':
        dataset = Reddit(root='data/Reddit', transform=NormalizeFeatures())
    if dataname == 'WikiCS':
        dataset = WikiCS(root='data/WikiCS', transform=NormalizeFeatures())   
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    return data,dataset

if __name__ == '__main__':
    pass

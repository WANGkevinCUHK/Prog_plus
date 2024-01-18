import torch
import torchmetrics
from torch_geometric.loader import DataLoader
from ProG.Model.model import GNN
from ProG.prompt import GPF,GPF_plus,LightPrompt
from torch import nn, optim
from ProG.Data.data import load_graph_task
from ProG.get_args import get_graph_task_args

def prompt_train(PG, train_loader, model, opi_pg, device, epoch, prompt_epoch):
    
    # training stage
    PG.train()
    
    for batch_id, train_batch in enumerate(train_loader):
        # print(train_batch)
        train_batch = train_batch.to(device)
        emb0 = model(train_batch.x, train_batch.edge_index, train_batch.batch)
        pg_batch = PG.inner_structure_update()
        pg_batch = pg_batch.to (device)
        pg_emb = model(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
        # cross link between prompt and input graphs
        dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
        
        sim = torch.softmax(dot, dim=1)

        train_loss = criterion(sim, train_batch.y)
        opi_pg.zero_grad()
        train_loss.backward()
        opi_pg.step()
        print('epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(epoch, prompt_epoch, batch_id+1, len(train_loader), train_loss))

def acc_f1_over_batches(test_loader, PG, model, num_class, device):
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    for batch_id, test_batch in enumerate(test_loader):
        test_batch = test_batch.to(device)
        emb0 = model(test_batch.x, test_batch.edge_index, test_batch.batch)
        pg_batch = PG.token_view()
        pg_batch = pg_batch.to(device)
        pg_emb = model(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
        dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
        pre = torch.softmax(dot, dim=1)

        y = test_batch.y
        pre_cla = torch.argmax(pre, dim=1)
        # print(pre_cla)
        # print(y)

        acc = accuracy(pre_cla, y)
        ma_f1 = macro_f1(pre_cla, y)
        print("Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id, acc.item(), ma_f1.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    print("Final True Acc: {:.4f} | Macro-F1: {:.4f}".format(acc.item(), ma_f1.item()))
    accuracy.reset()
    macro_f1.reset()
  
def train(model,train_loader,prompt,device):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch, prompt = prompt)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(model,loader, prompt, device):
    model.eval()

    correct = 0
    for data in loader: 
        data = data.to(device) # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch, prompt = prompt)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    args = get_graph_task_args()
    dataset, train_dataset, test_dataset = load_graph_task(args.dataset_name)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("prepare data is finished!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(input_dim=dataset.num_features,out_dim=dataset.num_classes, gnn_type = args.gnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    if args.prompt_type == 'None':
        prompt_type = None
    elif args.prompt_type =='ProG':
        lr, wd = 0.001, 0.00001
        prompt = LightPrompt(token_dim=dataset.num_features, token_num_per_group=100, group_num=dataset.num_classes, inner_prune=0.01)
        for p in model.parameters():
            p.requires_grad = False
        opi = optim.Adam(filter(lambda p: p.requires_grad, prompt.parameters()),lr=lr, weight_decay=wd)
    elif args.prompt_type == 'gpf':
        prompt = GPF(dataset.num_features).to(device)
    elif args.prompt_type == 'gpf-plus':
        prompt = GPF_plus(dataset.num_features,dataset.num_nodes).to(device)
    else:
        raise KeyError(" We don't support this kind of prompt.")


    if prompt_type == 'ProG':
        prompt_epoch = 200
        for j in range(1, prompt_epoch + 1):
            prompt_train(prompt, train_loader, model, opi, device, epoch = j, prompt_epoch = prompt_epoch)
            acc_f1_over_batches(test_loader, prompt, model, dataset.num_classes, device)

    else:
        for i in range(1, args.epochs +1):
            train(model, train_loader, prompt = prompt, device = device)
            train_acc = test(model, train_loader, prompt = prompt, device = device)
            test_acc = test(model, test_loader, prompt = prompt, device = device)
            print(f'Epoch: {i:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}') 
        

        

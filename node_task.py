
from ProG.Model.model import GNN, GPPT
import torch
from ProG.prompt import GPF,GPF_plus
from ProG.Data.data import load_node_task
from ProG.utils import constraint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name ='Cora'
data, dataset = load_node_task(dataset_name)
data = data.to(device)
data.train_id = torch.nonzero(data.train_mask).squeeze(1)
model = GNN(input_dim=dataset.num_features,out_dim=dataset.num_classes, gnn_type='GCN').to(device)

# setting prompt
prompt_type = 'gppt'
if prompt_type == 'gpf':
      prompt = GPF(data.num_features).to(device)
elif prompt_type == 'gpf-plus':
      prompt = GPF_plus(data.num_features,data.num_nodes ).to(device)
elif prompt_type == 'gppt':
      model = GPPT(in_feats = dataset.num_features, n_classes= dataset.num_classes).to(device)
      model.weigth_init(data.x,data.edge_index,data.y,data.train_id)
else:
      prompt = None

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train(model, data):
      # model.weigth_init(data.x,data.edge_index,data.y,data.train_id)
      if prompt_type != 'gppt':
            model.train()
            optimizer.zero_grad() 
            out = model(data.x, data.edge_index, batch=None, prompt = prompt) 
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  
            loss.backward()  
            optimizer.step()  
            return loss
      else:
            model.train()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss = loss + 0.001 * constraint(device,model.get_mul_prompt())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_prompt_weight(model.get_mid_h())
            return loss

      

def test(model, data, mask):
      model.eval()
      if prompt_type != 'gppt':
            out = model(data.x, data.edge_index, batch=None, prompt = prompt)
      else:
            out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  
      correct = pred[mask] == data.y[mask]  
      acc = int(correct.sum()) / int(mask.sum())  
      return acc


for epoch in range(1, 180):
    data = data.to(device)
    loss = train(model,data)
    val_acc = test(model,data,data.val_mask)
    test_acc = test(model,data,data.test_mask)
    print("Epoch {:03d} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, loss.item(), val_acc, test_acc))
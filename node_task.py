from ProG.Model.model import GNN, GPPT
import torch
from ProG.prompt import GPF,GPF_plus
from ProG.Data.data import load_node_task
from ProG.utils import constraint
from ProG.get_args import get_node_task_args

def train(model, data):
      # model.weigth_init(data.x,data.edge_index,data.y,data.train_id)
      if args.prompt_type != 'gppt':
            model.train()
            optimizer.zero_grad() 
            out = model(data.x, data.edge_index, batch=None, prompt = prompt) 
            loss = criterion(out[data.train_mask], data.y[data.train_mask])  #这里只用了train_mask的标签
            loss.backward()  
            optimizer.step()  
            return loss
      else:
      # if args.prompt_type is gppt prompt
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
      if args.prompt_type != 'gppt':
            out = model(data.x, data.edge_index, batch=None, prompt = prompt)
      else:
      # if args.prompt_type is gppt prompt
            out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  
      correct = pred[mask] == data.y[mask]  
      acc = int(correct.sum()) / int(mask.sum())  
      return acc

if __name__ == '__main__':
      args=get_node_task_args()

      device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
      data, dataset = load_node_task(args.dataset_name)
      data = data.to(device)
      data.train_id = torch.nonzero(data.train_mask).squeeze(1)
      model = GNN(input_dim=dataset.num_features,out_dim=dataset.num_classes, gnn_type=args.gnn_type).to(device)

      # load pre_trained model parameters

      # pre_train_path = '/home/chenyizi/ZCY/data/Prog_plus/pre_trained_gnn/{}.GraphCL.{}.pth'.format(args.dataset_name, args.gnn_type)
      # model.load_state_dict(torch.load(pre_train_path))
      # print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))

      # setting prompt
      if args.prompt_type == 'gpf':
            prompt = GPF(data.num_features).to(device)
      elif args.prompt_type == 'gpf-plus':
            prompt = GPF_plus(data.num_features,data.num_nodes ).to(device)
      elif args.prompt_type == 'gppt':
            model = GPPT(in_feats = dataset.num_features, n_classes= dataset.num_classes).to(device)
            model.weigth_init(data.x,data.edge_index,data.y,data.train_id)
      elif args.prompt_type == 'None':
            prompt = None
      else:
            raise KeyError(" We only support gpf, gpf-plus, gppt (or None) prompt typre for node task .")

      optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
      criterion = torch.nn.CrossEntropyLoss()

      for epoch in range(1, args.epochs):
            data = data.to(device)
            loss = train(model,data)
            val_acc = test(model,data,data.val_mask)
            test_acc = test(model,data,data.test_mask)
            print("Epoch {:03d} | Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, loss.item(), val_acc, test_acc))

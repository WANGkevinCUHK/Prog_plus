from torch_geometric.data import Data
from torch.utils.data import TensorDataset as TDataset
from torch.utils.data import DataLoader as TDataLoader
from utils import prepare_link_prediction_data
from torch.optim import Adam
import time
import torch


class LinkPredictionPretrain(PreTrain):
    def __init__(self, pretext_config=None, gnn_config=None, pre_train_graph_data=Data()):
        super(LinkPredictionPretrain, self).__init__(pretext_config, gnn_config, pre_train_graph_data)

        self.optimizer = Adam(self.gnn_enc.parameters(), lr=pretext_config["lr"], weight_decay=pretext_config["wd"])

    def generate_loader_data(self):
        edge_label, edge_index = prepare_link_prediction_data(self.graph_data)
        edge_index = edge_index.transpose(0, 1)

        data = TDataset(edge_label, edge_index)
        return TDataLoader(data, batch_size=self.pretext_config["batch_size"], shuffle=True)

    def pretrain_one_epoch(self):
        self.gnn_enc.train()

        accum_loss, total_step = 0, 0
        device = self.pretext_config["device"]

        criterion = torch.nn.BCEWithLogitsLoss()

        dataloader = self.generate_loader_data()
        for step, (batch_edge_label, batch_edge_index) in enumerate(dataloader):
            self.optimizer.zero_grad()

            batch_edge_label = batch_edge_label.to(device)
            batch_edge_index = batch_edge_index.to(device)

            node_emb = self.gnn_enc(self.graph_data.x.to(device), self.graph_data.edge_index.to(device))
            # TODO: should we re-design the Decoder?
            batch_pred_log = (node_emb[batch_edge_index[:, 0]] * node_emb[batch_edge_index[:, 1]]).sum(dim=-1)

            loss = criterion(batch_pred_log, batch_edge_label)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1

        return accum_loss / total_step

    def pretrain(self):
        num_epoch = self.pretext_config["epoch"]

        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()

            print(f"[Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")

        # TODO: save model
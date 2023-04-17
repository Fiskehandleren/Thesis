import argparse
import torch.optim
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from pytorch_lightning import LightningModule
import numpy as np
from utils.losses import _get_loss_metrics



class ATGCN(LightningModule):
    def __init__(
        self,
        loss_fn,
        edge_index,
        edge_weight,
        node_features: int,
        sequence_length: int,
        hidden_dim: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        censored= False,
        no_self_loops=False,
        **kwargs
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.censored = censored
        # Hyperparameters
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.no_self_loops = no_self_loops

        ATGCN._get_loss_metrics = _get_loss_metrics

        # We add improved self-loops for each node, to make sure that the nodes are weighing themselves
        # more than their neighbors. `improved=True` means that A_hat = A + 2I, so the diagonal is 3.
        self.atgcn = A3TGCN2(node_features, self.hidden_dim, periods=sequence_length, add_self_loops=not self.no_self_loops, improved=True, batch_size=batch_size)
        self.linear = torch.nn.Linear(hidden_dim, 1)

        # To save predictions and their true values for visualizations
        self.test_y = np.empty((0, 8))
        self.test_y_hat = np.empty((0, 8))

        self.save_hyperparameters(ignore=["model", "loss_fn", "edge_index", "edge_weight"])

    def forward(self, x, edge_index, edge_weight):
        h = self.atgcn(x, edge_index, edge_weight)
        y = F.relu(h)
        y = self.linear(h)
        return y.exp(), h
    
    def _get_preds_loss_metrics(self, batch, stage):
        x = batch[0]
        # Transfer graph stuff to device
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        
        # Make predictions
        y_hat, _ = self(x, self.edge_index, self.edge_weight)
        y_hat = y_hat.view(-1, x.shape[1])

        return self._get_loss_metrics(batch, y_hat, stage)
    
    def training_step(self, batch, batch_idx):
        loss_metrics, _, _= self._get_preds_loss_metrics(batch, "train")
        #self.log_dict(loss_metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss_metrics["train_loss"]
    
    def validation_step(self, batch, batch_idx):
        loss_metrics, _, _ = self._get_preds_loss_metrics(batch, "val")
        self.log_dict(loss_metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss_metrics["val_loss"]
    
    def test_step(self, batch, batch_idx):
        loss_metrics, y, y_hat = self._get_preds_loss_metrics(batch, "test")
        self.log_dict(loss_metrics, on_epoch=True, on_step=False, prog_bar=True)
        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate((self.test_y_hat, y_hat.cpu().detach().numpy()))

        return loss_metrics["test_loss"]
    
    def training_epoch_end(self, outputs) -> None:
        loss = np.mean([output['loss'].cpu().numpy() for output in outputs])
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser
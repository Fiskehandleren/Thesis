import argparse
import torch.optim
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.losses import get_loss

class TGCN_task(pl.LightningModule):
    def __init__(
        self,
        model,
        loss: str,
        edge_index,
        edge_weight,
        batch_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        censored= False,
        **kwargs
    ):
        super(TGCN_task, self).__init__()
        self.save_hyperparameters(ignore=["model", "loss", "edge_index", "edge_weight"])
        self.loss_func = get_loss(loss)
        self.model = model
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.censored = censored

    def training_step(self, batch, batch_idx):
        if self.censored:
            x, y, tau = batch
        else:
            x, y = batch
        y_hat, _ = self.model(x, self.edge_index, self.edge_weight)
        y_hat = y_hat.view(-1, 8)
        
        if self.censored:
            loss = self.loss_func(y_hat, y, tau)
        else:
            loss = self.loss_func(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        metrics = {
            "train_loss": loss,
            "train_mae": mae,
            "train_mse": mse
        }
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.censored:
            x, y, tau = batch
        else:
            x, y = batch
        y_hat, _ = self.model(x, self.edge_index.to(self.device), self.edge_weight.to(self.device))
        y_hat = y_hat.view(-1, 8)
        if self.censored:
            loss = self.loss_func(y_hat, y, tau)
        else:
            loss = self.loss_func(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        metrics = {
            "val_loss": loss,
            "val_mae": mae,
            "val_mse": mse
        }
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if self.censored:
            x, y, tau = batch
        else:
            x, y = batch
        y_hat, _ = self.model(x, self.edge_index.to(self.device), self.edge_weight.to(self.device))
        y_hat = y_hat.view(-1, 8)
        if self.censored:
            loss = self.loss_func(y_hat, y, tau)
        else:
            loss = self.loss_func(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        metrics = {
            "test_loss": loss,
            "test_mae": mae,
            "test_mse": mse
        }
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        return parser
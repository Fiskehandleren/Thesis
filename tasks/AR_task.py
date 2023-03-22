from torch import nn
import pytorch_lightning as pl
from torch import optim
import argparse 

class AR_Task(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        loss_fn,
        censored,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.censored = censored
        self._loss_fn = loss_fn
        self.feat_max_val = feat_max_val
        self.fc1 = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)
        return out.exp()
    
    def training_step(self, batch, batch_idx):
        # Run forward calculation
        if self.censored == True:
            x, y, tau = batch
        else:
            x, y = batch

        y_predict = self(x).view(-1)

        # Compute loss.
        if self.censored == True:
            loss = self._loss_fn(y_predict, y, tau)
        else:
            loss = self._loss_fn(y_predict, y)

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--output_dim", type=int, default=8)
        return parser
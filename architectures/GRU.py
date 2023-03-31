from torch import nn
import pytorch_lightning as pl
from torch import optim
import argparse 

class GRU(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim, 
        loss_fn,
        censored,
        hidden_units,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_layers = num_layers
        self.censored = censored
        self._loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feat_max_val = feat_max_val
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=hidden_units, out_features=output_dim)
        

    def forward(self, x):
        batch_size = x.shape[0]
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        
        #_, (hn, _) = self.gru(x, (h0, c0))
        x = x.view(batch_size, -1)
        _, (hn, _) = self.gru(x)
        out = self.linear(hn[-1]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=72)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--output_dim", type=int, default=1)
        return parser


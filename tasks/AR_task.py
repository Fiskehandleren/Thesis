from torch import nn
import pytorch_lightning as pl
from torch import optim

class AR_Task(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        censored,
        regressor="linear",
        pred_len: int = 2*24,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(AR_Task, self).__init__()
        self.save_hyperparameters()
        self.censored = censored
        self.model = model
        self._loss_fn = loss_fn
        self.feat_max_val = feat_max_val
        print(self.censored)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Run forward calculation
        if self.censored == True:
            x, y, tau = batch
        else:
            x, y = batch

        y_predict = self(x)

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
    
from ..losses import poisson_negative_log_likelihood
import pytorch_lightning as pl
from torch import optim

class AR_Task(pl.LightningModule):
    def __init__(
        self,
        model,
        regressor="linear",
        loss_fn=poisson_negative_log_likelihood,
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(AR_Task, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self._loss_fn = loss_fn
        self.feat_max_val = feat_max_val

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Run forward calculation
        x, y = batch
        y_predict = self(x)

        # Compute loss.
        loss = self._loss_fn(y_predict, y)

        # self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
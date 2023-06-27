import argparse
import numpy as np
from pytorch_lightning import LightningModule
import torch
import utils.constants
from utils.losses import get_loss_metrics, calculate_losses


class GraphTemporalBaseClass(LightningModule):
    def __init__(
        self,
        loss_fn,
        edge_index,
        edge_weight,
        node_features: int,
        forecast_horizon: int,
        sequence_length: int,
        hidden_dim: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        censored=False,
        no_self_loops=False,
        use_activation=False,
        use_dropout=False,
        print_cluster_loss=False,
        train_edge_weight=False,
        **kwargs,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        # Graph information
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.node_features = node_features

        self.censored = censored

        # Model features
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length

        # Debugging
        self.print_cluster_loss = print_cluster_loss

        # Hyperparameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.no_self_loops = no_self_loops
        self.use_activation = use_activation
        self.use_dropout = use_dropout
        self.train_edge_weight = train_edge_weight

        # Setup loss function
        GraphTemporalBaseClass.get_loss_metrics = get_loss_metrics

        # To save predictions and their true values for visualizations
        self.test_y = np.empty((0, 8, forecast_horizon))
        self.test_y_hat = np.empty((0, 8, forecast_horizon))
        self.test_y_true = np.empty((0, 8, forecast_horizon))

        self.save_hyperparameters(ignore=["loss_fn", "edge_index", "node_features"])

    def _get_preds_loss_metrics(self, batch, stage):
        y_hat = self._get_preds(batch)

        if self.print_cluster_loss and stage in ["val", "test"]:
            # Get loss for each node
            _, y, tau, y_true = batch
            for i in range(len(utils.constants.cluster_names)):
                loss, loss_true, mae, mse, rmse = calculate_losses(
                    y_hat[:, i, :],
                    y[:, i, :],
                    tau[:, i, :],
                    y_true[:, i, :],
                    self.censored,
                    self.loss_fn,
                )
                node_loss_metrics = {
                    f"{utils.constants.cluster_names[i]}_{stage}_loss": loss,
                    f"{utils.constants.cluster_names[i]}_{stage}_loss_true": loss_true,
                    f"{utils.constants.cluster_names[i]}_{stage}_mae": mae,
                    f"{utils.constants.cluster_names[i]}_{stage}_mse": mse,
                    f"{utils.constants.cluster_names[i]}_{stage}_rmse": rmse,
                }
                self.log_dict(node_loss_metrics, on_epoch=True, on_step=False)

        return self.get_loss_metrics(batch, y_hat, stage)

    def _get_preds(self, batch):
        x = batch[0]  # [b, 8, forecast_horizon, sequence_length]
        # Transfer graph stuff to device
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device)
        # Make predictions
        y_hat, _ = self(x, self.edge_index, self.edge_weight)
        return y_hat

    def training_step(self, batch, batch_idx):
        loss_metrics, _, _, _ = self._get_preds_loss_metrics(batch, "train")
        self.log_dict(loss_metrics, prog_bar=True)
        return loss_metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        loss_metrics, _, _, _ = self._get_preds_loss_metrics(batch, "val")
        self.log_dict(loss_metrics, on_epoch=True)
        return loss_metrics["val_loss"]

    def test_step(self, batch, batch_idx):
        loss_metrics, y, y_true, y_hat = self._get_preds_loss_metrics(batch, "test")
        self.log_dict(loss_metrics, on_epoch=True, on_step=False, prog_bar=True)

        self.test_y = np.concatenate((self.test_y, y.cpu().detach().numpy()))
        self.test_y_hat = np.concatenate(
            (self.test_y_hat, y_hat.cpu().detach().numpy())
        )
        if self.censored:
            self.test_y_true = np.concatenate(
                (self.test_y_true, y_true.cpu().detach().numpy())
            )
        return loss_metrics["test_loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument(
            "--no_self_loops",
            action="store_true",
            default=False,
            help="Censor data at cap. tau",
        )
        parser.add_argument(
            "--use_activation",
            action="store_true",
            default=False,
            help="Use ReLu after convolutional layer",
        )
        parser.add_argument(
            "--use_dropout",
            action="store_true",
            default=False,
            help="Use dropout after T-GCN layer",
        )
        parser.add_argument(
            "--print_cluster_loss",
            action="store_true",
            default=False,
            help="Print loss metrics for each cluster",
        )

        parser.add_argument(
            "--train_edge_weight",
            action="store_true",
            default=False,
            help="Train edge weights",
        )
        return parser

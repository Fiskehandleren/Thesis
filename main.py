import logging
import argparse
import pytorch_lightning as pl
from datasets import EVChargersDataset
from utils.losses import poisson_negative_log_likelihood
import models
from tasks import AR_Task, TGCN_task
from torch import nn

logger = logging.getLogger('Thesis.Train')

def get_model(args, dm):
    model = None
    if args.model_name == "AR":
        model = models.AR(input_dim=dm.seq_len, output_dim=2*24)
    elif args.model_name == "AR_Net":
        model = models.AR_Net(input_dim=dm.seq_len, output_dim=dm.pred_len, hidden_dim=args.hidden_dim)
    elif args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    #if args.model_name == "TGCN":
       #model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model

def get_loss(args):
    if args.loss == "mse":
        return nn.MSELoss()
    elif args.loss == "PNLL":
        return poisson_negative_log_likelihood

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("AR", "AR_Net", "DeepAR", "GRU"),
        default="AR",
    )

    parser.add_argument("--loss", type=str, help="Loss function to use", default="PNLL", choices=("mse", "PNLL"))

    temp_args, _ = parser.parse_known_args()
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)

    parser = EVChargersDataset.add_data_specific_arguments(parser)

    args = parser.parse_args()

    # Print arguments
    print(args)
    dm = EVChargersDataset(feat_path='data', adj_path="data", **vars(args))
    model = get_model(args, dm)
    if args.model_name == "GRU":
        task = TGCN_task(model, regressor="linear", **vars(args))
    else:
        loss_fn = get_loss(args)
        task = AR_Task(model, 'loss_fn', **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)

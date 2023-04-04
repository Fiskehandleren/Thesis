import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import numpy as np
import pandas as pd

import datasets
import architectures
from utils.losses import get_loss
from architectures import AR, TGCN, LSTM, GRU, ARNet

def get_model(args, dm):
    model = None
    loss_fn = get_loss(args.loss)

    if args.model_name == "TGCN":
        if args.censored:
            assert args.loss == "CPNLL", "Censored data only works with CPNLL loss. Rerun with --loss CPNLL"
        model = TGCN(edge_index=dm.edge_index, edge_weight=dm.edge_weight, node_features=dm.X_train.shape[2], loss_fn = loss_fn, **vars(args))
    elif args.model_name == "AR":
        assert not args.covariates, "AR models cannot include covariates"
        model = AR(input_dim=args.sequence_length, output_dim=1, loss_fn = loss_fn, **vars(args))
    elif args.model_name == "ARNet":
        assert not args.covariates, "AR models cannot include covariates"
        model = ARNet(input_dim=args.sequence_length, output_dim=1, loss_fn = loss_fn, **vars(args))
    elif args.model_name == "LSTM":
        model = LSTM(input_dim=dm.input_dimensions, output_dim=1, loss_fn = loss_fn, **vars(args))
    elif args.model_name == "GRU":
        model = GRU(input_dim=dm.input_dimensions, loss_fn = loss_fn, **vars(args))
    else:
        raise ValueError(f"{args.model_name} not implemented yet!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, help="The name of the model", 
        choices=("AR", "ARNet", "LSTM", "TGCN", "GRU"), required=True)
    
    parser.add_argument("--dataloader", type=str, help="Name of dataloader", choices=("EVChargersDatasetSpatial", "EVChargersDataset"), required = True)

    parser.add_argument("--loss", type=str, help="Loss function to use", default="PNLL", choices=("MSE", "PNLL", "CPNLL", "CPNLL_TGCN"))

    # Common dataset arguments
    parser.add_argument("--cluster", type=str, help="Which cluster to fit model to")
    parser.add_argument("--covariates", help="Add covariates to the dataset", default=False, action='store_true')
    parser.add_argument("--censored", action='store_true', default = False, help= "Censor data at cap. tau")
    parser.add_argument("--censor_level", default = 1, help = "Choose censorship level")
    parser.add_argument("--forecast_lead", type=int, default=24, help="How many time steps ahead to predict")
    parser.add_argument("--sequence_length",  type=int, default = 72)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_start", type=str, required=True)
    parser.add_argument("--train_end", type=str, required=True)
    parser.add_argument("--test_end", type=str, required=True)



    temp_args, _ = parser.parse_known_args()
    #parser = getattr(architectures, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(datasets, temp_args.dataloader).add_data_specific_arguments(parser)
    parser = getattr(architectures, temp_args.model_name).add_model_specific_arguments(parser)
    args = parser.parse_args()

    dm = getattr(datasets, temp_args.dataloader)(**vars(args))

    # Print arguments
    print(args)
    
    model = get_model(args, dm)

    wandb.login()

    wandb_logger = WandbLogger(project='Thesis', log_model='all')
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='max')
    
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    trainer.save_checkpoint(f"trained_models/best_model_{args.model_name}_{args.loss}.ckpt")

    # pd.DataFrame(trainer.callback_metrics).to_csv(f"trained_models/best_model_{args.model_name}_{args.loss}.csv")

    #if (args.model_name == "TGCN" or args.model_name == "LSTM" or args.model_name == "GRU" or args.model_name == "AR" or args.model_name == "ARNet"):
    # TODO implement test-step for rest of the models
    trainer.test(model, datamodule=dm)
    wandb.finish()
    df_dates = pd.DataFrame(dm.y_dates, columns=['Date'])
    df_true = pd.DataFrame(model.test_y, columns=dm.cluster_names)
    df_pred = pd.DataFrame(model.test_y_hat, columns=np.char.add(dm.cluster_names, '_pred'))
    pd.concat([df_dates, df_true, df_pred], axis=1).to_csv(f"predictions/predictions_{args.model_name}_{args.loss}_{args.hidden_dim}_{args.censor_level}.csv")

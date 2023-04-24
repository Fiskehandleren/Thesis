import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import gc
import numpy as np
import torch
from os import remove    

import datasets
import architectures
from utils.losses import get_loss
from utils.plotting_functions import generate_prediction_html, generate_prediction_data
from architectures import AR, TGCN, LSTM, GRU, ARNet, ATGCN

def get_trained_model(args, dm):
    artifact_dir = args.pretrained
    # If we're loading an artifact from wandb, we need to download it first
    if ":" in args.pretrained:
        run = wandb.init(job_type='predict', )
        artifact = run.use_artifact(artifact_dir, type='model')
        artifact_dir = artifact.download() + '/model.ckpt'
    if args.model_name == 'TGCN':
        model = getattr(architectures, temp_args.model_name).load_from_checkpoint(artifact_dir, edge_index=dm.edge_index, edge_weight=dm.edge_weight, loss_fn = get_loss(args.loss))
    else:
        model = getattr(architectures, temp_args.model_name).load_from_checkpoint(artifact_dir, loss_fn = get_loss(args.loss))
    return model

def get_model(args, dm):
    model = None

    if args.pretrained:
        return get_trained_model(args, dm)

    loss_fn = get_loss(args.loss)

    if args.model_name == "TGCN":
        if args.censored:
            assert args.loss == "CPNLL", "Censored data only works with CPNLL loss. Rerun with --loss CPNLL"
        model = TGCN(edge_index=dm.edge_index, edge_weight=dm.edge_weight, node_features=dm.X_train.shape[1], loss_fn = loss_fn, **vars(args))
    elif args.model_name == "ATGCN":
        if args.censored:
            assert args.loss == "CPNLL", "Censored data only works with CPNLL loss. Rerun with --loss CPNLL"
        model = ATGCN(edge_index=dm.edge_index, edge_weight=dm.edge_weight, node_features=dm.X_train.shape[1], loss_fn = loss_fn, **vars(args))
    elif args.model_name == "AR":
        assert not args.covariates, "AR models cannot include covariates"
        model = AR(input_dim=args.sequence_length, output_dim=1, loss_fn = loss_fn, **vars(args))
    elif args.model_name == "ARNet":
        assert not args.covariates, "AR models cannot include covariates"
        model = ARNet(input_dim=args.sequence_length, output_dim=1, loss_fn = loss_fn, **vars(args))
    elif args.model_name == "LSTM":
        model = LSTM(input_dim=dm.input_dimensions, output_dim=1, loss_fn = loss_fn, **vars(args))
    elif args.model_name == "GRU":
        model = GRU(input_dim=dm.input_dimensions, output_dim=1, loss_fn = loss_fn, **vars(args))
    else:
        raise ValueError(f"{args.model_name} not implemented yet!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--mode", choices=("train", "test", "predict"), default="train")
    parser.add_argument("--model_name", type=str, help="The name of the model", 
        choices=("AR", "ARNet", "LSTM", "TGCN", "GRU", "ATGCN"), required=True)
    
    parser.add_argument("--dataloader", type=str, help="Name of dataloader", choices=("EVChargersDatasetSpatial", "EVChargersDataset"), required = True)
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model", default=None)
    parser.add_argument("--loss", type=str, help="Loss function to use", default="PNLL", choices=("MSE", "PNLL", "CPNLL", "CPNLL_TGCN"))

    # Common dataset arguments
    parser.add_argument("--cluster", type=str, help="Which cluster to fit model to")
    parser.add_argument("--covariates", help="Add covariates to the dataset", default=False, action='store_true')
    parser.add_argument("--censored", action='store_true', default = False, help= "Censor data at cap. tau")
    parser.add_argument("--censor_level", default = 1, help = "Choose censorship level")
    parser.add_argument("--censor_dynamic", default = False, help = "Use dynamic censoring scheme", action='store_true')
    parser.add_argument("--forecast_lead", type=int, default=24, help="How many time steps ahead to predict")
    parser.add_argument("--sequence_length",  type=int, default = 72)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_start", type=str, required=True)
    parser.add_argument("--train_end", type=str, required=True)
    parser.add_argument("--test_end", type=str, required=True)
    parser.add_argument("--val_end", type=str, required=False)


    temp_args, _ = parser.parse_known_args()
    parser = getattr(datasets, temp_args.dataloader).add_data_specific_arguments(parser)
    parser = getattr(architectures, temp_args.model_name).add_model_specific_arguments(parser)
    args = parser.parse_args()

    dm = getattr(datasets, temp_args.dataloader)(**vars(args))

    # Print arguments
    print(args)
    
    model = get_model(args, dm)
    
    wandb_logger = WandbLogger(project='Thesis', log_model='all', job_type=args.mode)
    #wandb_logger.watch(model)


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_last=True)
    
    run_name = wandb.run.name
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[checkpoint_callback])
    predictions = None
    if args.mode == "train":
        trainer.fit(model, dm, ckpt_path=args.pretrained)
        trainer.test(model, datamodule=dm)
        # Save local model
        trainer.save_checkpoint(f"trained_models/best_model_{run_name}.ckpt")
        predictions = generate_prediction_data(dm, model)
        html_path = generate_prediction_html(predictions, run_name)
        # We might want to save metrics locally
        # pd.DataFrame(trainer.callback_metrics).to_csv(f"trained_models/best_model_{args.model_name}_{args.loss}.csv")
        wandb.log({"test_predictions": wandb.Html(open(html_path), inject=False)})
        remove(html_path)
    elif args.mode == 'predict':
        trainer.predict(model, datamodule=dm, return_predictions=False)
        predictions = generate_prediction_data(dm, model)

    predictions.to_csv(f"predictions/predictions_{args.model_name}_{run_name}.csv")
    wandb.finish()

    del model
    del dm
    del trainer
    del predictions
    
    gc.collect()
    torch.cuda.empty_cache()

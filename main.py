import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import gc
import pandas as pd
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
        assert (
            args.logger == True
        ), "If you're loading a model from wandb, you must use the wandb logger"
        run = wandb.init(
            job_type="predict",
        )
        artifact = run.use_artifact(artifact_dir, type="model")
        artifact_dir = artifact.download() + "/model.ckpt"
    if args.model_name == "TGCN":
        model = getattr(architectures, temp_args.model_name).load_from_checkpoint(
            artifact_dir,
            edge_index=dm.edge_index,
            edge_weight=dm.edge_weight,
            loss_fn=get_loss(args.loss),
            node_features=dm.X_train.shape[1],
        )
    else:
        model = getattr(architectures, temp_args.model_name).load_from_checkpoint(
            artifact_dir, loss_fn=get_loss(args.loss)
        )
    return model


def get_model(args, dm):
    model = None

    if args.pretrained:
        return get_trained_model(args, dm)

    loss_fn = get_loss(args.loss)

    if args.model_name == "TGCN":
        model = TGCN(
            edge_index=dm.edge_index,
            edge_weight=dm.edge_weight,
            node_features=dm.X_train.shape[1],
            loss_fn=loss_fn,
            **vars(args),
        )
    elif args.model_name == "ATGCN":
        model = ATGCN(
            edge_index=dm.edge_index,
            edge_weight=dm.edge_weight,
            node_features=dm.X_train.shape[1],
            loss_fn=loss_fn,
            **vars(args),
        )
    elif args.model_name == "AR":
        model = AR(
            input_dim=args.sequence_length, output_dim=1, loss_fn=loss_fn, **vars(args)
        )
    elif args.model_name == "ARNet":
        model = ARNet(input_dim=args.sequence_length, loss_fn=loss_fn, **vars(args))
    elif args.model_name == "LSTM":
        model = LSTM(input_dim=dm.input_dimensions, loss_fn=loss_fn, **vars(args))
    elif args.model_name == "GRU":
        model = GRU(input_dim=dm.input_dimensions, loss_fn=loss_fn, **vars(args))
    else:
        raise ValueError(f"{args.model_name} not implemented yet!")
    return model


def validate_args(parser):
    args = parser.parse_args()

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    test_end = pd.Timestamp(args.test_end)

    if train_start >= train_end:
        parser.error("Training start date must be before training end date")

    if train_end >= test_end:
        parser.error("Training end date must be before test end date")

    if not test_end > val_end > train_end:
        parser.error(
            "Test end date must be after validation end date, which must be after training end date"
        )

    if args.loss == "PNLL" and args.censored:
        parser.error("PNLL loss cannot be used with censoring")

    if args.covariates and ("AR" in args.model_name):
        parser.error("AR models cannot include covariates")

    if not args.logger and args.save_predictions:
        parser.error("If you're saving predictions, you must use a logger")

    return args


if __name__ == "__main__":
    print("Starting at: ", pd.Timestamp.now())
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = Trainer.add_argparse_args(parser)

    # Model and data related arguments
    parser.add_argument("--mode", choices=("train", "test", "predict"), default="train")
    parser.add_argument(
        "--save_predictions",
        help="Store predictions after training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model",
        choices=("AR", "ARNet", "LSTM", "TGCN", "GRU", "ATGCN"),
        required=True,
    )

    parser.add_argument(
        "--dataloader",
        type=str,
        help="Name of dataloader",
        choices=("EVChargersDatasetSpatial", "EVChargersDataset"),
        required=True,
    )
    parser.add_argument(
        "--pretrained", type=str, help="Path to pretrained model", default=None
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to use",
        default="PNLL",
        choices=("MSE", "PNLL", "CPNLL", "CPNLL_TGCN"),
    )

    # Common dataset arguments
    parser.add_argument("--cluster", type=str, help="Which cluster to fit model to")
    parser.add_argument(
        "--covariates",
        help="Add covariates to the dataset",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--censored", action="store_true", default=False, help="Censor data at cap. tau"
    )
    parser.add_argument("--censor_level", default=1, help="Choose censorship level")
    parser.add_argument(
        "--censor_dynamic",
        default=False,
        help="Use dynamic censoring scheme",
        action="store_true",
    )
    parser.add_argument(
        "--forecast_lead",
        type=int,
        default=1,
        help="How many time steps ahead to predict",
    )
    parser.add_argument(
        "--forecast_horizon", type=int, default=1, help="How many time steps to predict"
    )
    parser.add_argument("--sequence_length", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_start", type=str, required=True)
    parser.add_argument("--train_end", type=str, required=True)
    parser.add_argument("--test_end", type=str, required=True)
    parser.add_argument("--val_end", type=str, required=False)

    # Parse known arguments to get dataloader and model name
    temp_args, _ = parser.parse_known_args()
    parser = getattr(datasets, temp_args.dataloader).add_data_specific_arguments(parser)
    parser = getattr(architectures, temp_args.model_name).add_model_specific_arguments(
        parser
    )

    # Validate and parse arguments
    args = validate_args(parser)

    if "T00:00:00Z" in args.train_start:
        args.train_start = args.train_start.replace("T00:00:00Z", "")
    if "T00:00:00Z" in args.train_end:
        args.train_end = args.train_end.replace("T00:00:00Z", "")
    if "T00:00:00Z" in args.test_end:
        args.test_end = args.test_end.replace("T00:00:00Z", "")
    if "T00:00:00Z" in args.val_end:
        args.val_end = args.val_end.replace("T00:00:00Z", "")

    # Initialize datamodule
    dm = getattr(datasets, temp_args.dataloader)(**vars(args))

    # Print arguments
    print(args)

    # Setup logger
    if args.logger:
        import wandb

        wandb_logger = WandbLogger(
            project="Thesis", log_model="all", job_type=args.mode
        )
        run_name = wandb.run.id
    else:
        wandb_logger = None
        run_name = "local"

    # Initialize model
    model = get_model(args, dm)

    # Setup checkpoint
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    # Initialize trainer
    trainer = Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=[checkpoint_callback]
    )

    # Train, test, and predict
    predictions = []
    if args.mode == "train":
        trainer.fit(model, dm, ckpt_path=args.pretrained)
        trainer.test(model, datamodule=dm, ckpt_path="best")
        # Save local model
        # trainer.save_checkpoint(f"trained_models/best_model_{run_name}.ckpt")
        if args.save_predictions:
            predictions = generate_prediction_data(dm, model)
            for tup in predictions:
                cluster, prediction = tup[0], tup[1]
                if args.logger:
                    html_path = generate_prediction_html(prediction, run_name)
                    wandb.log(
                        {
                            f"test_predictions_{cluster}": wandb.Html(
                                open(html_path), inject=False
                            )
                        }
                    )
                    remove(html_path)
    elif args.mode == "predict":
        trainer.test(model, datamodule=dm, ckpt_path="best")
        predictions = generate_prediction_data(dm, model)

    # Log predictions
    for tup in predictions:
        cluster, prediction = tup[0], tup[1]
        prediction.to_csv(
            f"predictions/predictions_{args.model_name}_{cluster}_{run_name}_{args.censor_level}.csv"
        )
        html_path = generate_prediction_html(prediction, run_name)
        wandb.log(
            {f"test_predictions_{cluster}": wandb.Html(open(html_path), inject=False)}
        )
        remove(html_path)

    if args.logger:
        wandb.finish()

    del model
    del dm
    del trainer
    del predictions

    gc.collect()
    torch.cuda.empty_cache()

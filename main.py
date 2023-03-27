import logging
import argparse
import pytorch_lightning as pl
import pandas as pd

import architectures
import datasets
from utils.losses import get_loss
from tasks import AR_Task, TGCN_task, LSTM_task, GRU_task

logger = logging.getLogger('Thesis.Train')

def get_model(args, dm):
    model = None
    if args.model_name == "AR":
        model = architectures.AR(input_dim=args.sequence_length, output_dim=1)
    elif args.model_name == "AR_Net":
        model = architectures.AR_Net(input_dim=dm.sequence_length, output_dim=dm.pred_len, hidden_dim=args.hidden_dims)
    elif args.model_name == "LSTM": 
        model = architectures.LSTM(input_dim=dm.input_dimensions, hidden_units=args.hidden_dim)
    elif args.model_name == "GRU":
        model = architectures.GRU(input_dim=dm.input_dimensions, hidden_units=args.hidden_dim)
    elif args.model_name == "TemporalGCN":
       model = architectures.TemporalGCN(node_features=dm.X_train.shape[2], hidden_dim=args.hidden_dim, time_steps=args.lags, batch_size=args.batch_size)
    elif model is None:
        raise ValueError("Model not found") 
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, help="The name of the model for spatiotemporal prediction", 
        choices=("AR", "AR_Net", "LSTM", "TemporalGCN", "GRU"), required=True)
    
    parser.add_argument("--dataloader", type=str, help="Name of dataloader", choices=("EVChargersDataset", "EVChargersDatasetLSTM"), required = True)

    parser.add_argument("--loss", type=str, help="Loss function to use", default="PNLL", choices=("mse", "PNLL", "CPNLL", "CPNLL_TGCN"))

    temp_args, _ = parser.parse_known_args()
    parser = getattr(architectures, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(datasets, temp_args.dataloader).add_data_specific_arguments(parser)

    args = parser.parse_args()

    dm = getattr(datasets, temp_args.dataloader)(**vars(args))

    # Print arguments
    print(args)
    
    model = get_model(args, dm)

    if args.censored:
        assert args.loss == "CPNLL", "Censored data only works with CPNLL loss. Rerun with --loss CPNLL"
    
    if args.model_name == "TemporalGCN":
        task = TGCN_task(model, edge_index=dm.edge_index, edge_weight=dm.edge_weight, **vars(args))
    
    elif (args.model_name == "AR" or args.model_name == "AR_Net"):
        assert not args.covariates, "AR models cannot include covariates" 

        loss_fn = get_loss(args.loss)
        task = AR_Task(input_dim=args.sequence_length, output_dim=1, loss_fn = loss_fn, **vars(args))
    
    elif(args.model_name == "LSTM"):
        loss_fn = get_loss(args.loss)
        task = LSTM_Task(input_dim=args.sequence_length, output_dim = 1, loss_fn = loss_fn, **vars(args))

    elif(args.model_name == "GRU"):
        loss_fn = get_loss(args.loss)
        task = GRU_Task(input_dim=args.sequence_length, output_dim = 1, loss_fn = loss_fn, **vars(args))
    
    '''
    else:
        if (args.model_name == "AR" or args.model_name == "AR_Net"):
            assert not args.covariates, "AR models cannot include covariates"
        loss_fn = get_loss(args.loss)
        
        if (args.model_name == "AR" or args.model_name == "AR_Net"):
            task = AR_Task(input_dim=args.sequence_length, output_dim=1, loss_fn = loss_fn, **vars(args))
    '''

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)
    trainer.save_checkpoint(f"trained_models/best_model_{args.model_name}_{args.loss}.ckpt")
    trainer.test(task, datamodule=dm)

    # pd.DataFrame(trainer.callback_metrics).to_csv(f"trained_models/best_model_{args.model_name}_{args.loss}.csv")
    
    if args.model_name == "TemporalGCN":
        df_true = pd.DataFrame(task.test_y, columns=dm.cluster_names)
        df_pred = pd.DataFrame(task.test_y_hat, columns=dm.cluster_names + '_pred')
        pd.concat([df_true, df_pred], axis=1).to_csv(f"predictions/predictions_{args.model_name}_{args.loss}.csv")
import logging
import argparse
import pytorch_lightning as pl
from utils.losses import get_loss

#from datasets import EVChargersDataset, EVChargersDatasetLSTM
import models
import datasets
from tasks import AR_Task, TGCN_task

logger = logging.getLogger('Thesis.Train')

def get_model(args, dm):
    model = None
    if args.model_name == "AR":
        model = models.AR(input_dim=dm.seq_len, output_dim=2*24)
    elif args.model_name == "AR_Net":
        model = models.AR_Net(input_dim=dm.seq_len, output_dim=dm.pred_len, hidden_dim=args.hidden_dim)
    elif args.model_name == "LSTM": 
        model = models.LSTM(input_dim=dm.input_dimensions, hidden_units = args.hidden_dim)
    elif args.model_name == "TemporalGCN":
       model = models.TemporalGCN(node_features=dm.X_train.shape[2], hidden_dim=args.hidden_dim, time_steps=args.lags, batch_size=args.batch_size)
    elif model is None:
        raise ValueError("Model not found") 
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, help="The name of the model for spatiotemporal prediction", 
        choices=("AR", "AR_Net", "DeepAR", "LSTM", "TemporalGCN"), required=True)
    
    parser.add_argument("--dataloader", type=str, help="Name of dataloader", choices=("EVChargersDataset", "EVChargersDatasetLSTM"), required = True)

    parser.add_argument("--loss", type=str, help="Loss function to use", default="PNLL", choices=("mse", "PNLL"))

    temp_args, _ = parser.parse_known_args()
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(datasets, temp_args.dataloader).add_data_specific_arguments(parser)

    #parser = temp_args.dataloader.add_data_specific_arguments(parser)
    args = parser.parse_args()

    dm = getattr(datasets, temp_args.dataloader)(**vars(args))

    '''
    if temp_args.model_name == "LSTM":
        parser = EVChargersDatasetLSTM.add_data_specific_arguments(parser)
        dm = EVChargersDatasetLSTM(**vars(args))
    else:
        dm = EVChargersDataset(feat_path='data', **vars(args))
    '''

    # Print arguments


    print(args)
    
    model = get_model(args, dm)

    if args.model_name == "TemporalGCN":
        task = TGCN_task(model, edge_index=dm.edge_index, edge_weight=dm.edge_weight, **vars(args))
    else:
        loss_fn = get_loss(args)
        task = AR_Task(model, **vars(args))
        
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(task, dm)

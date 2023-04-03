from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import utils.dataloader as dataloader
import argparse


class EVChargersDataset(pl.LightningDataModule):
    def __init__(
        self,
        covariates: bool,
        batch_size: int,
        forecast_lead: int,
        censored: bool,
        censor_level: int,
        cluster: str,
        multiple_stations: bool,
        sequence_length: int,
        train_start: str,
        train_end: str,
        test_end: str,
        #val_end: str,
        #val_end: str,
        **kwargs
    ):
        
        super().__init__()
        self.covariates = covariates
        self.batch_size = batch_size
        self.forecast_lead = forecast_lead
        self.cluster = cluster
        self.censored = censored
        self.censor_level = censor_level
        self.cluster_names =  np.array([cluster])
        if self.censored:
            self.tau = cluster + '_TAU' 
        else:
            self.tau = None
        self.train_start = train_start
        self.train_end = train_end
        self.test_end = test_end
        self.multiple_stations = multiple_stations
        self.sequence_length = sequence_length
    
        self.df_train, self.df_test, self.features, self.target = dataloader.get_datasets_NN(target = self.cluster, forecast_lead = self.forecast_lead, add_month=self.covariates, 
                                                                                                add_hour = self.covariates, add_day_of_week=self.covariates, add_year = self.covariates,
                                                                                                train_start = self.train_start, train_end = self.train_end, test_end = self.test_end,
                                                                                                is_censored = self.censored,
                                                                                                multiple_stations=self.multiple_stations, censorship_level = self.censor_level)

        self.input_dimensions = len(self.features)

    def train_dataloader(self):
        train_dataset = dataloader.SequenceDataset(self.df_train, self.target, self.features, self.tau, self.sequence_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 8)

    def test_dataloader(self):
        test_dataset = dataloader.SequenceDataset(self.df_test, self.target, self.features, self.tau, self.sequence_length)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 8)
    

    '''
    def setup(self, stage=None):
        self.train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train), torch.FloatTensor(self.y_train)
        )
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_test), torch.FloatTensor(self.y_test)
        )
    '''


    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--covariates", help="Add covariates to the dataset", type=bool, default=False)
        parser.add_argument("--cluster", type=str, help="Which cluster to fit model to", default = 'WEBSTER')
        parser.add_argument("--forecast_lead", type=int, default=24)
        parser.add_argument("--censored", action='store_true', default = False, help= "Censor data at cap. tau")
        parser.add_argument("--censor_level", default = 1, help = "Choose censorship level")
        parser.add_argument("--train_start", type=str, required=True)
        parser.add_argument("--train_end", type=str, required=True)
        parser.add_argument("--test_end", type=str, required=True)
        parser.add_argument("--sequence_length",  type=int, default = 72)
        parser.add_argument("--multiple_stations", action='store_true', default = False, help="Include data from all stations for prediction")

        return parser

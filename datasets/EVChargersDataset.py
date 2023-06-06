from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import argparse

import utils.dataloader as dataloader


class EVChargersDataset(pl.LightningDataModule):
    def __init__(
        self,
        covariates: bool,
        batch_size: int,
        forecast_lead: int,
        forecast_horizon: int,
        censored: bool,
        censor_level: int,
        censor_dynamic: bool,
        cluster: str,
        multiple_stations: bool,
        sequence_length: int,
        train_start: str,
        train_end: str,
        test_end: str,
        val_end: str,
        **kwargs
    ):
        super().__init__()
        self.covariates = covariates
        self.batch_size = batch_size
        self.forecast_lead = forecast_lead
        self.forecast_horizon = forecast_horizon
        self.cluster = cluster
        self.censored = censored
        self.censor_level = censor_level
        self.censor_dynamic = censor_dynamic
        self.cluster_names = np.array([cluster])
        self.tau = cluster + "_TAU"
        self.true_target = cluster + "_TRUE"

        self.train_start = train_start
        self.train_end = train_end
        self.test_end = test_end
        self.val_end = val_end
        self.multiple_stations = multiple_stations
        self.sequence_length = sequence_length

        (
            self.df_train,
            self.df_test,
            self.df_val,
            self.features,
            self.target,
        ) = dataloader.get_datasets_NN(
            target=self.cluster,
            forecast_lead=self.forecast_lead,
            censor_dynamic=self.censor_dynamic,
            add_month=self.covariates,
            add_hour=self.covariates,
            add_day_of_week=self.covariates,
            add_year=self.covariates,
            train_start=self.train_start,
            train_end=self.train_end,
            test_end=self.test_end,
            val_end=self.val_end,
            multiple_stations=self.multiple_stations,
            censorship_level=self.censor_level,
        )

        self.input_dimensions = len(self.features)
        # Drop the first sequence_length samples, as these are used as input to the model
        self.y_dates = self.df_test.iloc[sequence_length:].Period.to_numpy()

    def train_dataloader(self):
        train_dataset = dataloader.SequenceDataset(
            self.df_train,
            self.target,
            self.features,
            self.tau,
            self.true_target,
            self.sequence_length,
            forecast_horizon=self.forecast_horizon,
        )
        return DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def test_dataloader(self):
        test_dataset = dataloader.SequenceDataset(
            self.df_test,
            self.target,
            self.features,
            self.tau,
            self.true_target,
            self.sequence_length,
            forecast_horizon=self.forecast_horizon,
        )
        return DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def val_dataloader(self):
        val_dataset = dataloader.SequenceDataset(
            self.df_val,
            self.target,
            self.features,
            self.tau,
            self.true_target,
            self.sequence_length,
            forecast_horizon=self.forecast_horizon,
        )
        return DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--multiple_stations",
            action="store_true",
            default=False,
            help="Include data from all stations for prediction",
        )

        return parser

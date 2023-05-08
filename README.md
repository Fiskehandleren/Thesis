# Thesis
[![Test models](https://github.com/Fiskehandleren/Thesis/actions/workflows/models.yml/badge.svg)](https://github.com/Fiskehandleren/Thesis/actions/workflows/models.yml)

## How to train TGCN
```bash
python main.py --model_name TemporalGCN \
    ---train_start 2018-01-01 --train_end 2019-01-01 --val_end 2019-05-02 --test_end 2019-06-30 \
    --covariates --batch_size 32 --max_epochs 10 \
    --censored --dataloader EVChargersDatasetSpatial --loss CPNLL
```

## How to train censored AR
```bash
python main.py --model_name AR \
    --train_start 2019-01-01 --train_end 2019-05-01 --val_end 2019-05-02 --test_end 2019-06-30 \
    --batch_size 32 --max_epochs 10 --dataloader EVChargersDataset  --censored --loss CPNLL
```

## Predict
`--pretrained` takes either a model path from Wandb or a local path to a model checkpoint. When running `--mode predict` you can pass the config of the dataset you want to model to run on. Here the sequence length and forecast lead should match the model. The model will predict on the test period.
```bash 
python main.py --mode predict --pretrained fiskehandleren/Thesis/model-232ybnqc:v1 \
    --model_name TGCN \
    --train_start 2018-01-01 --train_end 2019-01-01 --val_end 2019-05-02 --test_end 2019-06-30 \
    --covariates --batch_size 32  --dataloader EVChargersDatasetSpatial\
    --loss CPNLL --forecast_lead 1 --censor_level 2
```


## Note on censored datasets:
``` charging_session_count_1_to_30_censored_1.csv: observations capped at value 1
# charging_session_count_1_to_30_censored_2.csv: observations capped at value 2
# charging_session_count_1_to_30_censored_4.csv: observations capped at 2 below maximum number of plugs (when maximum #                                                number of plugs is equal or above 4)
# charging_session_count_1_to_30_censored_5.csv: observations capped at 1 below maximum number of plugs (when maximum   #                                                number of plugs is equal or above 4)´´´


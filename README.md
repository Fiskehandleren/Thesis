# Thesis

## How to train TGCN
```bash
python main.py --model_name TemporalGCN --train_start 2019-01-01 --train_end 2019-05-01 --test_end 2020-05-30 --val_start 2019-04-01 --val_end 2019-04-30 --covariates --batch_size 32 --max_epochs 10 --censored --dataloader EVChargersDatasetSpatial --loss CPNLL
```

## How to train censored AR
```bash
python main.py --model_name AR --train_start 2019-01-01 --train_end 2019-05-01 --test_start 2020-05-02 --batch_size 32 --max_epochs 10 --dataloader EVChargersDataset --test_end 2020-05-30 --censored --loss CPNLL
```


## Note on censored datasets:
# charging_session_count_1_to_30_censored_1.csv: observations capped at value 1
# charging_session_count_1_to_30_censored_2.csv: observations capped at value 2
# charging_session_count_1_to_30_censored_4.csv: observations capped at 2 below maximum number of plugs (when maximum #                                                number of plugs is equal or above 4)
# charging_session_count_1_to_30_censored_5.csv: observations capped at 1 below maximum number of plugs (when maximum   #                                                number of plugs is equal or above 4)


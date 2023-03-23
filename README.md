# Thesis

## How to train TGCN
```bash
python main.py --model_name TemporalGCN --train_start 2019-01-01 --train_end 2019-05-01 --test_start 2020-05-01 --test_end 2020-05-30 --val_start 2019-04-01 --val_end 2019-04-30 --covariates --batch_size 32 --max_epochs 10 --censored --dataloader EVChargersDataset --loss CPNLL
```

## How to train censored AR
```bash
python main.py --model_name AR --train_start 2019-01-01 --train_end 2019-05-01 --test_start 2020-05-02 --batch_size 32 --max_epochs 10 --dataloader EVChargersDatasetLSTM --test_end 2020-05-30 --censored --loss CPNLL
```
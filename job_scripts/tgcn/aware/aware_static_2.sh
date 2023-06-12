#!/bin/sh
### General options https://www.hpc.dtu.dk/?page_id=1519
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J TGCN_cpnll_static_2_lead_48_covariates
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=15GB]"
#BSUB -u s203331@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
#BSUB -o /dev/null
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --
bash python3 main.py --loss=CPNLL --mode=train \
--devices=1 --val_end=2019-04-01 --censored --test_end=2019-07-01 --train_end=2018-10-01 \
--batch_size=128 --dataloader=EVChargersDatasetSpatial --hidden_dim=512 \
--max_epochs=15 --model_name=TGCN --accelerator=gpu --train_start=2017-01-01 \
--weight_decay=0.0001 --forecast_lead=48 \
--learning_rate=0.0001 --sequence_length=336 --forecast_horizon=1 \
--save_predictions \
--censor_level=2 --adjecency_threshold=0 --use_activation --covariates

## ONLY CHANGE HYPERPERAMS - NOT CENSORLEVEL OR DYNAMIC

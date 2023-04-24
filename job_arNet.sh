#!/bin/sh
### General options
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J arNet
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 04:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s174045@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo job_out/gpu-%J.out
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --

## CPNLL
## bash python3 -m wandb wandb agent latend-demand/Thesis/ry4tql5x

## PNLL
## bash python3 -m wandb agent latend-demand/Thesis/gjwfrtmw

## bash python3 main.py --model_name ARNet --cluster WEBSTER --train_start 2019-01-01 --train_end 2019-05-01 --test_end 2019-06-01 --val_end 2019-06-20 --sequence_length 215 --hidden_dim 53 --batch_size 16 --max_epochs 1 --learning_rate 1e-2 --weight_decay 0.0721 --dataloader EVChargersDataset --censored --loss CPNLL --censor_level 2 --censor_dynamic


## bash python3 main.py --model_name ARNet --cluster WEBSTER --train_start 2019-01-01 --train_end 2019-05-01 --test_end 2019-06-01 --val_end 2019-06-20 --batch_size 32 --max_epochs 4 --dataloader EVChargersDataset --censored --loss CPNLL --censor_level 3 --covariates False --accelerator gpu --devices 1
## bash python3 main.py --model_name ARNet --cluster WEBSTER --train_start 2019-01-01 --train_end 2019-05-01 --test_end 2019-06-01 --val_end 2019-06-20 --batch_size 32 --max_epochs 2 --dataloader EVChargersDataset --loss PNLL --censor_level 3 --accelerator gpu --devices 1

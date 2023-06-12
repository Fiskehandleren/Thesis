#!/bin/sh
### General options
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ar_stat2_1_48
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s203331@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /dev/null
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --

## CPNLL
## bash python3 -m wandb wandb agent latend-demand/Thesis/ry4tql5x

## PNLL
## bash python3 -m wandb agent latend-demand/Thesis/gjwfrtmw

## bash python3 main.py --model_name ARNet --cluster WEBSTER --train_start 2019-01-01 --train_end 2019-05-01 --test_end 2019-06-01 --val_end 2019-06-20 --sequence_length 215 --hidden_dim 53 --batch_size 16 --max_epochs 1 --learning_rate 1e-2 --weight_decay 0.0721 --dataloader EVChargersDataset --censored --loss CPNLL --censor_level 2 --censor_dynamic

CLUSTERS=("BRYANT" "MPL" "CAMBRIDGE" "RINCONADA" "HAMILTON" "TED" "HIGH" "WEBSTER")
n=${#CLUSTERS[@]}
for j in 1 2 3 4
  do
  for i in $(seq 0 "$(($n-1))")
    do
    bash python3 main.py --loss=CPNLL --censored --mode=train --logger --cluster "${CLUSTERS[$i]}" --devices=1 \
    --max_steps=-1 --num_nodes=1 --precision=32 \
    --train_start=2017-01-01 --val_end=2019-04-01 --test_end=2019-07-01 --train_end=2018-10-01 \
    --batch_size=32 --dataloader=EVChargersDataset --max_epochs=20 \
    --model_name=AR --accelerator=gpu --censor_level=2 --weight_decay=0.06470644058196104 \
    --forecast_lead=48 --learning_rate=0.0005927787748490534 --inference_mode --sequence_length=336 \
    --forecast_horizon=1
  done
done
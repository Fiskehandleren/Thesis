#!/bin/sh
### General options
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J lstm_cpnll_all_clusters
### -- ask for number of cores (default: 1) --
#BSUB -n 8
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
#BSUB -o /dev/null
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --

## CPNLL
## bash python3 -m wandb agent latend-demand/Thesis/7c5279ww

## bash python3 main.py --model_name LSTM --cluster WEBSTER --train_start 2018-01-01 --train_end 2019-01-01 --val_end 2019-05-02 --test_end 2019-06-30 --sequence_length 303 --hidden_dim 97 --batch_size 17 --max_epochs 10 --save_predictions --learning_rate 0.0005 --weight_decay 0.02526 --dataloader EVChargersDataset  --loss CPNLL --censor_level 2 --censored --accelerator gpu --devices 1


## PNLL
## bash python3 -m wandb agent latend-demand/Thesis/mlc4opyr
## bash python3 main.py --model_name LSTM --cluster WEBSTER --train_start 2018-01-01 --train_end 2019-01-01 --val_end 2019-06-30 --test_end 2019-05-02 --sequence_length 303 --hidden_dim 97 --batch_size 17 --max_epochs 15 --learning_rate 0.01931 --weight_decay 0.02526 --dataloader EVChargersDataset --loss PNLL --censor_level 2 --censor_dynamic --accelerator gpu --devices 1


CLUSTERS=("BRYANT" "MPL" "CAMBRIDGE" "RINCONADA" "HAMILTON" "TED" "HIGH" "WEBSTER")
n=${#CLUSTERS[@]}

for i in $(seq 0 "$(($n-1))")
  do
  bash python3 main.py --loss=CPNLL --mode=train --logger --cluster "${CLUSTERS[$i]}" --devices=1 --val_end=2019-05-02 \
  --censored --test_end=2019-06-30 --max_steps=-1 --num_nodes=1 --precision=32 \
  --train_end=2019-01-01 --batch_size=16 --covariates --dataloader=EVChargersDataset --hidden_dim=44 --max_epochs=30 \
  --model_name=LSTM --num_layers=1 --accelerator=gpu --train_start=2017-01-01 --censor_level=3 --weight_decay=0.06622145460281466 \
  --forecast_lead=1 --learning_rate=0.0005661019045297409 --inference_mode --sequence_length=336 --track_grad_norm=-1 \
  --forecast_horizon=1 --save_predictions --log_every_n_steps=50 --enable_progress_bar --replace_sampler_ddp --enable_checkpointing \
  --enable_model_summary --num_sanity_val_steps=2 --check_val_every_n_epoch=1 --multiple_trainloader_mode=max_size_cycle 
 done

#!/bin/sh
### General options
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J gru_cpnll
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

CLUSTERS=("BRYANT" "MPL" "CAMBRIDGE" "RINCONADA" "HAMILTON" "TED" "HIGH" "WEBSTER")
n=${#CLUSTERS[@]}

for i in $(seq 0 "$(($n-1))")
  do
  bash python3 main.py --loss=CPNLL --mode=train --logger --cluster "${CLUSTERS[$i]}"  --devices=1 --val_end=2019-05-02 \
  --censored --test_end=2019-06-30 --max_steps=-1 --num_nodes=1 --precision=32 \
  --train_end=2019-01-01 --batch_size=16 --dataloader=EVChargersDataset --hidden_dim=232 --max_epochs=30 \
  --model_name=GRU --num_layers=1 --accelerator=gpu --train_start=2017-01-01 --censor_level=2 --weight_decay=0.0013216782040014494 \
  --forecast_lead=1 --learning_rate=0.0009184290390652538 --censor_dynamic --inference_mode --sequence_length=336 --track_grad_norm=-1 \
  --forecast_horizon=1 --save_predictions --log_every_n_steps=50 --enable_progress_bar --replace_sampler_ddp --enable_checkpointing \
  --enable_model_summary --num_sanity_val_steps=2 --check_val_every_n_epoch=1 --multiple_trainloader_mode=max_size_cycle
 done

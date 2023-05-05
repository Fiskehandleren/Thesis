#!/bin/sh
### General options https://www.hpc.dtu.dk/?page_id=1519
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J TGCN_sweep_pnll
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=10GB]"
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
## BSUB -oo job_out/gpu_%J.out
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --
# bash python3 main.py --loss=CPNLL --mode=train --logger --devices=1 --val_end=2019-05-02 --censored --test_end=2019-06-30 --max_steps=-1 --num_nodes=1 --precision=32 --train_end=2019-01-01 --batch_size=67 --dataloader=EVChargersDatasetSpatial --hidden_dim=38 --max_epochs=10 --model_name=TGCN --accelerator=gpu --train_start=2017-01-01 --censor_level=2 --forecast_lead=1 --learning_rate=0.0001 --no_self_loops --censor_dynamic --inference_mode --sequence_length=336 --track_grad_norm=-1 --forecast_horizon=1 --log_every_n_steps=50 --enable_progress_bar --replace_sampler_ddp --enable_checkpointing --enable_model_summary --num_sanity_val_steps=2 --check_val_every_n_epoch=1 --multiple_trainloader_mode=max_size_cycle
bash python3 -m wandb agent latend-demand/Thesis/d1d5dyx1
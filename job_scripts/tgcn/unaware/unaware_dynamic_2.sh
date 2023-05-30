#!/bin/sh
### General options https://www.hpc.dtu.dk/?page_id=1519
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J TGCN_cpnll_dynamic_2_unaware_lead_1
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
bash python3 main.py --loss=PNLL --mode=train \
--devices=1 --val_end=2019-04-01 --test_end=2019-07-01 --train_end=2018-10-01 \
--batch_size=256 --dataloader=EVChargersDatasetSpatial --hidden_dim=408 \
--max_epochs=10 --model_name=TGCN --accelerator=gpu --train_start=2017-01-01 \
--weight_decay=0.0008445547076417635 --forecast_lead=1 \
--learning_rate=0.0007437002957287736 --sequence_length=336 --forecast_horizon=1 \
--log_every_n_steps=50 --enable_progress_bar --replace_sampler_ddp --enable_checkpointing \
--enable_model_summary --num_sanity_val_steps=2 --check_val_every_n_epoch=1 \
--multiple_trainloader_mode=max_size_cycle --save_predictions \
--censor_level=2 --censor_dynamic

## ONLY CHANGE HYPERPERAMS - NOT CENSORLEVEL OR DYNAMIC

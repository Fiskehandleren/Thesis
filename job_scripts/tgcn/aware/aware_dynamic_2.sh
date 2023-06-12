#!/bin/sh
### General options https://www.hpc.dtu.dk/?page_id=1519
### - specify queue --
### classstat 
### check if gpu is available on queue with `nodestat -g <queue_name>`
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J TGCN_cpnll_dynamic_2_lead_1_adjecency_1.5_lower_wd_1024_dim
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=20GB]"
#BSUB -u s203331@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
#BSUB -o /dev/null
#BSUB -eo job_out/gpu_%J.err
# -- end of LSF options --
bash python3 main.py --loss=CPNLL --mode=predict --devices=1 --val_end=2019-04-01 --censored --test_end=2019-07-01 --train_end=2018-10-01 --batch_size=128 --dataloader=EVChargersDatasetSpatial --hidden_dim=700 --max_epochs=20 --model_name=TGCN --accelerator=gpu --train_start=2017-01-01 --weight_decay=0.0001 --forecast_lead=1 --learning_rate=0.0001 --sequence_length=336 --forecast_horizon=1 --censor_level=2 --censor_dynamic --adjecency_threshold=1.5  --use_activation --pretrained=latend-demand/Thesis/model-rz2mtv2o:v12 # --print_cluster_loss --use_dropout  --save_predictions 

## ONLY CHANGE HYPERPERAMS - NOT CENSORLEVEL OR DYNAMIC

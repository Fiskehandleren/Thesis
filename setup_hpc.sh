#!/bin/bash

#Open interactive GPU node
voltash

module load cuda/11.6
module load python3/3.8.2


python3 -m venv thesis

source thesis/bin/activate
# export PYTHONPATH='
pip install --upgrade pip

#pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric-temporal
pip install -r requirements.txt

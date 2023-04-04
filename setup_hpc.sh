#Open interactive node - Login to login2.hpc.dtu.dk & login3.hpc.dtu.dk

#ssh sXXXXXX@login3.hpc.dtu.dk
#Enter Password

#Open interactive GPU node
voltash

#Load preinstalled modules
module load cuda/11.6

# if no conda, follow this https://www.hpc.dtu.dk/?page_id=3678 

source $HOME/miniconda3/bin/activate
conda install python=
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric-temporal

pip install -r requirements.txt

echo "Done"
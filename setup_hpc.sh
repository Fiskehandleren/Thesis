#Open interactive node - Login to login2.hpc.dtu.dk & login3.hpc.dtu.dk

#ssh sXXXXXX@login3.hpc.dtu.dk
#Enter Password

#Open interactive GPU node
k40sh

#Load preinstalled modules
module load python3/3.9.14
module load gcc

#Create a virtual environment for Python3
python3 -m venv thesis

#Activate virtual environment
source ~/thesis/bin/activate

#If pip3 fails, use: which pip3, to make sure it is the one in the virutal environment.
#which pip3

pip3 install -r requirements.txt

#Panda for handling data
# module load pandas/0.20.3-python-3.6.2
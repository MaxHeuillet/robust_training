# robust_training

### Create the python environment:

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a
python3.11 -m venv ~/scratch/myenv_reprod
pip install -r requirements.txt

Note: the code runs with python 3.11.


### Before runing code:

Check the paths in the default configuration files, located in ./configs directory.

### To launch all the jobs on the cluster:



### To use in interactive session:

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.11 arrow/18.1.0 scipy-stack/2024a httpproxy
source ~/scratch/myenv_reprod/bin/activate
cd ./project_directory
bash ./dataset_to_tmpdir.sh 'uc-merced-land-use-dataset' 
python distributed_experiment_final.py

Note: the code runs a default configuration specified in ./utils/arguments.py:


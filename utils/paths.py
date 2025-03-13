import os

def get_data_dir(hp_opt, config):
    # Retrieve the node name
    nodename = os.uname().nodename.lower()
    
    # Define keywords to identify the cluster environment
    cluster_keywords = ["calculquebec", "calcul.quebec"]
    
    # Check if the node is part of the Calcul Québec cluster
    if any(keyword in nodename for keyword in cluster_keywords):
        # Retrieve the SLURM_TMPDIR environment variable
        # the data archive is send for dataset file to the TMPDIR for more efficiency
        slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
        
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR is not set. Please ensure you're running within a SLURM job.")
        
        # Construct the data directory path
        data_dir = os.path.join(slurm_tmpdir, 'data')
    else:
        # Define the default data directory for non-cluster environments
        # this is if you run prototypes locally
        data_dir = os.path.abspath('/home/mheuillet/Desktop/robust_training/data')
    
    return data_dir

    
def get_state_dict_dir(hp_opt,config):

    if "calculquebec" in os.uname().nodename or "calcul.quebec" in os.uname().nodename:  # Check for a substring that is unique to the cluster
        # this is to load state dict (not during HP opt), you can specify relative path to your state dict directory
        statedict_dir = '~/scratch/state_dicts_share/' #TO UPDATE
        if hp_opt:
            ### this is to load state dict during HP OPT, you must specify an absolute path to the directory
            statedict_dir = "/home/mheuill/scratch/state_dicts_share/" #TOUP
    else:
        # this is if you run prototypes locally
        statedict_dir = "/home/mheuillet/Desktop/robust_training/state_dicts/"
    return statedict_dir
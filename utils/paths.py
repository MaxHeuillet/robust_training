import os

def get_data_dir(hp_opt, config):
    # Retrieve the node name
    nodename = os.uname().nodename.lower()
    
    # Define keywords to identify the cluster environment
    cluster_keywords = ["calculquebec", "calcul.quebec"]
    
    # Check if the node is part of the Calcul Québec cluster
    if any(keyword in nodename for keyword in cluster_keywords):
        # Retrieve the SLURM_TMPDIR environment variable
        slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
        
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR is not set. Please ensure you're running within a SLURM job.")
        
        # Construct the data directory path
        data_dir = os.path.join(slurm_tmpdir, 'data')
    else:
        # Define the default data directory for non-cluster environments
        data_dir = os.path.abspath('/home/mheuillet/Desktop/robust_training/data')
    
    return data_dir

    
def get_state_dict_dir(hp_opt,config):

    if "calculquebec" in os.uname().nodename or "calcul.quebec" in os.uname().nodename:  # Check for a substring that is unique to the cluster
        statedict_dir = './state_dicts/'
        if hp_opt:
            statedict_dir = "/home/mheuill/projects/def-adurand/mheuill/robust_training/state_dicts/"
    else:
        statedict_dir = "/home/mheuillet/Desktop/robust_training/state_dicts/"
    return statedict_dir
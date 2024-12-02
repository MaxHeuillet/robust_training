import os

def get_data_dir(hp_opt,config):

    if "calculquebec" in os.uname().nodename or "calcul.quebec" in os.uname().nodename: 
        data_dir = '~/scratch/data'
        if hp_opt:
            data_dir = '/home/mheuill/scratch/data'
    else:
        data_dir = '/home/mheuillet/Desktop/robust_training/data'
    return data_dir

    
def get_state_dict_dir(hp_opt,config):

    if "calculquebec" in os.uname().nodename or "calcul.quebec" in os.uname().nodename:  # Check for a substring that is unique to the cluster
        statedict_dir = './state_dicts/'
        if hp_opt:
            statedict_dir = "/home/mheuill/projects/def-adurand/mheuill/robust_training/state_dicts/"
    else:
        statedict_dir = "/home/mheuillet/Desktop/robust_training/state_dicts/"
    return statedict_dir
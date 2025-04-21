# serialization_utils.py

import os
import pickle

from crptmidfreq.config_loc import get_feature_folder


def save_instance(instance):
    """
    Save a pickled instance to the specified folder.
    The file is saved as <folder>/<name>.pkl.
    """
    folder = instance.folder
    name=instance.name
    filepath = os.path.join(get_feature_folder(),folder, name + ".pkl")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(instance, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_instance(cls, folder, name,**kwargs):
    """
    Load an instance of the class from a pickle file.
    
    If the file doesn't exist, create a new instance using the provided parameters.
    Optionally, pass in a function (get_feature_folder_func) to compute the full folder path.
    """
    filepath = os.path.join(get_feature_folder(),folder, name + ".pkl")
    if not os.path.exists(filepath):
        return cls(folder=folder, name=name, **kwargs)
    with open(filepath, "rb") as f:
        instance = pickle.load(f)
    return instance

import os
import pickle

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.utils.common import validate_input
class BaseStepper:
    
    
    _instances = {}  # Class variable to track loaded instances
    verbose=3
    
    
    def __init__(self, folder='', name=''):
        self.folder = os.path.join(get_feature_folder(), folder)
        self.name = name

    def __hash__(self):
        # Use a tuple of the important attributes to compute the hash
        return hash((self.folder, self.name))
    
    def save_utility(self):
        """Save internal state to file."""
        # Ensure the folder exists
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        # Dynamically get the state of the instance by filtering out callable objects
        state = {}
        
        for key, value in self.__dict__.items():
            if not callable(value):
                if isinstance(value, Dict):  # Check if it's a numba Dict
                    # Convert numba Dict to a Python dict
                    state[key] = dict(value)
                else:
                    state[key] = value
        
        # Filepath where the state will be saved
        filepath = os.path.join(self.folder, self.name + '.pkl')

        # Save the state using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        

    @classmethod
    def load_utility(self,cls, folder='', name='',**kwargs):
        """Load instance from saved state or create new if not exists"""
        folder_path = os.path.join(get_feature_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            if self.verbose>0:
                print(f'Stepper creating instance {folder} {name}')
            return cls(folder=folder, name=name, **kwargs)

        if self.verbose>0:
            print(f'gStepper loading instance {folder} {name}')
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        instance = cls(folder=folder_path, name=name, **kwargs)

        # Iterate over all items in the state dictionary and create corresponding instance variables
        for key, value in state.items():
            if isinstance(value, dict):
                # Populate the dict from the saved state
                # TODO: clean this mess here
                #dict_value_type = str(getattr(instance, key)._numba_type_.value_type)
                for k, v in value.items():
                    #if 'array' in dict_value_type:
                    #    getattr(instance, key)[k] = np.array([v],dtype=np.float64)
                    #else:
                    getattr(instance, key)[k] = v
            else:
                # For non-dict keys like 'window'
                setattr(instance, key, value)
        return instance


    def validate_input(self, dt, dscode, serie=None,**kwargs):
        """
        Common input validation for update methods in subclasses.

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Raises:
            ValueError: If input validation fails.
        """
        validate_input(dt,dscode,serie=serie,**kwargs)
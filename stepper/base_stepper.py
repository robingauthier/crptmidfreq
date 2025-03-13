import os
import pickle
from config_loc import get_feature_folder


class BaseStepper:
    
    
    _instances = {}  # Class variable to track loaded instances
    verbose=3
    
    @classmethod
    def load_old(cls, folder: str, name: str, window: float):
        instance_key = f"{folder}/{name}"
        try:
            instance = cls._load_from_file(folder, name)
        except (FileNotFoundError, ValueError):
            instance = cls(folder=folder, name=name, window=window)
        
        cls._instances[instance_key] = instance
        return instance
    

    def load_utility(self,cls, folder='', name='',**kwargs):
        """Load instance from saved state or create new if not exists"""
        folder_path = os.path.join(get_feature_folder(), folder)
        filepath = os.path.join(folder_path, name + '.pkl')

        if not os.path.exists(filepath):
            if self.verbose>0:
                print(f'RollingStepper creating instance {folder} {name}')
            return cls(folder=folder, name=name, **kwargs)

        print(f'RollingStepper loading instance {folder} {name}')
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        instance = cls(folder=folder_path, name=name)

        # Iterate over all items in the state dictionary and create corresponding instance variables
        for key, value in state.items():
            if isinstance(value, dict):  # For nested dictionary types
                # For keys like 'values1', 'values2', etc., that are supposed to be numba Dicts
                setattr(instance, key, Dict.empty(
                    key_type=types.int64,
                    value_type=types.Array(types.float64, 1, 'C')
                ))
                # Populate the dict from the saved state
                for k, v in value.items():
                    getattr(instance, key)[k] = np.array(v)
            else:
                # For non-dict keys like 'window'
                setattr(instance, key, value)

        return instance
    
    @classmethod
    def _load_from_file(cls, folder: str, name: str):
        raise NotImplementedError("Subclasses must implement _load_from_file")
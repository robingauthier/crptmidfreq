class BaseStepper:
    _instances = {}  # Class variable to track loaded instances
    
    @classmethod
    def load(cls, folder: str, name: str, window: float):
        instance_key = f"{folder}/{name}"
        #if instance_key in cls._instances:
        #    return cls._instances[instance_key]
            
        try:
            instance = cls._load_from_file(folder, name)
        except (FileNotFoundError, ValueError):
            instance = cls(folder=folder, name=name, window=window)
        
        cls._instances[instance_key] = instance
        return instance
    
    @classmethod
    def _load_from_file(cls, folder: str, name: str):
        raise NotImplementedError("Subclasses must implement _load_from_file")
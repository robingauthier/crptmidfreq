import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types
import os
from .base_stepper import BaseStepper



class ModelPredStepper(BaseStepper):
    
    def __init__(self, folder='', name='', model=None):
        super().__init__(folder,name)
        self.model=model

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, window=1):
        """Load instance from saved state or create new if not exists"""
        return ModelPredStepper.load_utility(cls,folder=folder,name=name)

    def update(self, X):
        """
        """
        return self.model.predict(X)

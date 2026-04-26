# src_py/core/base_sim.py
# Phase 5: Abstract Base Class defining the mode interface

from abc import ABC, abstractmethod
import pandas as pd

class BaseSimulator(ABC):
    def __init__(self):
        self.dataset = None
        
    def load_data(self, df: pd.DataFrame):
        self.dataset = df
        
    @abstractmethod
    def define_target_variable(self) -> str:
        """ Returns the exact target variable name for the active mode """
        pass
        
    @abstractmethod
    def preprocess_features(self) -> tuple:
        """ Returns X (Features Tensor) and Y (Target Vector) """
        pass
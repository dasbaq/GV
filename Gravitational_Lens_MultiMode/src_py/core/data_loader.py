# src_py/core/data_loader.py
# Phase 4: Standardized CSV parser for C-Engine outputs

import pandas as pd
import os

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Missing Engine Output: {self.filepath}")

    def load_trajectory_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        return df
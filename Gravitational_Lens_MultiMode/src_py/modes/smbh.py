# src_py/modes/smbh.py

from core.base_sim import BaseSimulator

class SMBHMode(BaseSimulator):
    def define_target_variable(self) -> str:
        return "SMBH_Source_Reconstruction"
        
    def preprocess_features(self) -> tuple:
        if self.dataset is None:
            raise ValueError("데이터셋이 로드되지 않았습니다.")
            
        X = self.dataset[['final_x', 'final_y', 'time_delay']]
        Y = self.dataset[['init_x', 'init_y']]
        
        return X, Y
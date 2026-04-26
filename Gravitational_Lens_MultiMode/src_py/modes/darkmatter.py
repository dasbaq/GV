# src_py/modes/darkmatter.py

from core.base_sim import BaseSimulator

class DarkMatterMode(BaseSimulator):
    def define_target_variable(self) -> str:
        # 타겟 변수명을 목적에 맞게 수정
        return "Source_Position_Reconstruction"
        
    def preprocess_features(self) -> tuple:
        if self.dataset is None:
            raise ValueError("데이터셋이 로드되지 않았습니다.")
            
        # X (Feature): 지구에서 관측 가능한 렌즈 통과 후의 좌표와 시간 지연
        X = self.dataset[['final_x', 'final_y', 'time_delay']]
        
        # Y (Target): 머신러닝이 맞춰야 하는 빛의 원래 출발점 (다중 출력)
        Y = self.dataset[['init_x', 'init_y']]
        
        return X, Y
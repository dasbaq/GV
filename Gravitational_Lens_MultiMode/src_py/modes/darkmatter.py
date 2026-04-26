# src_py/modes/darkmatter.py
# Phase 9: NFW 프로파일 특화 피처 엔지니어링 추가

import numpy as np
from core.base_sim import BaseSimulator

class DarkMatterMode(BaseSimulator):
    def define_target_variable(self) -> str:
        # 타겟 변수명을 목적에 맞게 수정
        return "Source_Position_Reconstruction"
        
    def preprocess_features(self) -> tuple:
        if self.dataset is None:
            raise ValueError("데이터셋이 로드되지 않았습니다.")
        
        df = self.dataset.copy()
            
        # 기본 관측 피처: 지구에서 관측 가능한 렌즈 통과 후의 좌표와 시간 지연
        # 파생 피처: 극좌표 변환
        df['radial_r'] = np.sqrt(df['final_x']**2 + df['final_y']**2)
        df['polar_theta'] = np.arctan2(df['final_y'], df['final_x'])
        
        # NFW 프로파일 반영 파생 피처
        df['r_inv'] = 1.0 / (df['radial_r'] + 1e-6)
        df['nfw_profile'] = 1.0 / (df['radial_r'] * (1.0 + df['radial_r']) + 1e-6)
        df['log_delay'] = np.log1p(df['time_delay'])
        
        # 교차 피처
        df['x_times_delay'] = df['final_x'] * df['time_delay']
        df['y_times_delay'] = df['final_y'] * df['time_delay']
        
        # 2차 다항식 교차 피처 (비선형 렌즈 방정식 포착)
        df['x_squared'] = df['final_x'] ** 2
        df['y_squared'] = df['final_y'] ** 2
        df['xy_cross'] = df['final_x'] * df['final_y']
        
        # 암흑물질 서브구조 위치 (2.0, 1.5) 기준 거리
        df['dist_to_sub_x'] = df['final_x'] - 2.0
        df['dist_to_sub_y'] = df['final_y'] - 1.5
        df['dist_to_sub_r'] = np.sqrt(df['dist_to_sub_x']**2 + df['dist_to_sub_y']**2)
        
        # X (Feature)
        X = df[['final_x', 'final_y', 'time_delay',
                'radial_r', 'polar_theta', 'r_inv', 'nfw_profile',
                'log_delay', 'x_times_delay', 'y_times_delay',
                'x_squared', 'y_squared', 'xy_cross',
                'dist_to_sub_x', 'dist_to_sub_y', 'dist_to_sub_r']]
        
        # Y (Target): 머신러닝이 맞춰야 하는 빛의 원래 출발점 (다중 출력)
        Y = df[['init_x', 'init_y']]
        
        return X, Y
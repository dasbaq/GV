# src_py/modes/smbh.py
# Phase 9: Point Mass 특화 피처 엔지니어링 추가

import numpy as np
from core.base_sim import BaseSimulator

class SMBHMode(BaseSimulator):
    def define_target_variable(self) -> str:
        return "SMBH_Source_Reconstruction"
        
    def preprocess_features(self) -> tuple:
        if self.dataset is None:
            raise ValueError("데이터셋이 로드되지 않았습니다.")
        
        df = self.dataset.copy()
        
        # 기본 파생 피처: 극좌표 변환
        df['radial_r'] = np.sqrt(df['final_x']**2 + df['final_y']**2)
        df['polar_theta'] = np.arctan2(df['final_y'], df['final_x'])
        
        # Point Mass 특화 피처
        df['r_inv'] = 1.0 / (df['radial_r'] + 1e-6)
        df['r_inv_sq'] = df['r_inv'] ** 2  # 강한 렌즈 영역에서 중요
        df['log_delay'] = np.log1p(df['time_delay'])
        df['delay_over_r'] = df['time_delay'] / (df['radial_r'] + 1e-6)
        
        # 교차 피처
        df['x_times_delay'] = df['final_x'] * df['time_delay']
        df['y_times_delay'] = df['final_y'] * df['time_delay']
        
        # SMBH 중심(0,0) 근처의 극단적 편향 보정용
        df['log_r'] = np.log1p(df['radial_r'])
        
        # 2차 다항식 교차 피처 (비선형 렌즈 방정식 포착)
        df['x_squared'] = df['final_x'] ** 2
        df['y_squared'] = df['final_y'] ** 2
        df['xy_cross'] = df['final_x'] * df['final_y']
        
        # Point Mass 역문제 특화: 아인슈타인 반경 근사
        df['einstein_approx'] = np.sqrt(df['time_delay'] / (df['radial_r'] + 1e-6))
        
        # X (Feature)
        X = df[['final_x', 'final_y', 'time_delay',
                'radial_r', 'polar_theta', 'r_inv', 'r_inv_sq',
                'log_delay', 'delay_over_r',
                'x_times_delay', 'y_times_delay', 'log_r',
                'x_squared', 'y_squared', 'xy_cross',
                'einstein_approx']]
        
        # Y (Target)
        Y = df[['init_x', 'init_y']]
        
        return X, Y
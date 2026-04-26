from core.base_sim import BaseSimulator
import pandas as pd
import numpy as np
import scipy.signal
from core.lightcurve import LightCurveSimulator

class CosmologyMode(BaseSimulator):
    def define_target_variable(self) -> str:
        return "Time_Delay_from_Unresolved_Curve"
        
    def preprocess_features(self) -> tuple:
        if self.dataset is None:
            raise ValueError("데이터셋이 로드되지 않았습니다.")
            
        print("⏳ [Data Eng] 10,000개의 미분해 광도곡선 생성 및 ACF(자기상관) 피처 추출 중...")
        print("   (복잡한 시계열 계산으로 인해 약 10~20초 정도 소요됩니다.)")

        # 3일 간격으로 1000일 관측 가정
        t_obs = np.arange(0, 1000, 3.0)
        simulator = LightCurveSimulator(tau=150.0, sigma=0.5, mean_flux=20.0)
        
        X_features = []
        Y_targets = []
        
        # 10,000개의 C엔진 데이터에 대해 반복
        for _, row in self.dataset.iterrows():
            # 1. 퀘이사 원본 빛 생성
            source_flux = simulator.generate_drw_source(t_obs)
            
            # 2. C엔진의 물리량 가져오기 (Image A는 0에 도착, B는 delay 후 도착 가정)
            t_A = 0.0
            t_B = row['time_delay']
            
            # 3. 미분해 광도곡선 합성
            joint_flux, _, _ = simulator.generate_unresolved_curve(
                t_obs, source_flux, t_A, t_B, mu_A=1.2, mu_B=0.8, noise_level=0.02
            )
            
            # 4. ACF (자기상관함수) 계산
            n = len(joint_flux)
            var = np.var(joint_flux)
            flux_norm = joint_flux - np.mean(joint_flux)
            # numpy를 이용한 빠른 교차상관 계산 -> FFT 기반 O(N log N) 초고속 연산으로 교체
            acf = scipy.signal.correlate(flux_norm, flux_norm, mode='full', method='fft')[n-1:] / (var * n)
            
            # 처음 100개(약 300일 치)의 시차(Lag) 데이터만 피처로 사용
            X_features.append(acf[:100])
            Y_targets.append(t_B) # 예측해야 할 정답은 '진짜 시간 지연'
            
        # 100차원의 시계열 패턴이 X, 단일 시간 지연값이 Y
        X = pd.DataFrame(X_features)
        Y = pd.Series(Y_targets)
        
        return X, Y
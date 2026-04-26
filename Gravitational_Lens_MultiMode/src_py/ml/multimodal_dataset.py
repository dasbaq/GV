import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# 프로젝트 내의 다른 파이썬 파일을 불러오기 위한 경로 설정
sys.path.append(os.path.abspath('src_py')) 

# 선생님의 실제 광도곡선 시뮬레이터 불러오기
from core.lightcurve import LightCurveSimulator

class GravitationalLensDataset(Dataset):
    def __init__(self, spatial_csv, seq_length=100):
        self.spatial_df = pd.read_csv(spatial_csv)
        self.spatial_data = self.spatial_df[['init_x', 'init_y', 'final_x', 'final_y']].values
        self.spatial_data = np.array(self.spatial_data, dtype=np.float32)
        
        self.targets = self.spatial_df[['time_delay', 'true_H0']].values
        self.targets = np.array(self.targets, dtype=np.float32)
        
        self.seq_length = seq_length
        self.spatial_scaler = StandardScaler()
        self.spatial_data = self.spatial_scaler.fit_transform(self.spatial_data)
        
        # 💡 광도곡선 시뮬레이터 초기화 (한 번만 로드하여 재사용)
        self.lc_simulator = LightCurveSimulator(tau=100.0, sigma=0.2, mean_flux=10.0)

    def __len__(self):
        return len(self.spatial_df)

    def __getitem__(self, idx):
        # 1. 공간 데이터
        spatial_sample = torch.FloatTensor(self.spatial_data[idx])
        
        # 2. 진짜 광도곡선 데이터 즉석 생성
        time_delay = self.targets[idx][0] # csv에서 가져온 진짜 시간 지연 값
        
        try:
            # 관측 시간 배열 생성 (예: 1000일 동안 관측)
            t_obs = np.linspace(0, 1000, self.seq_length)
            
            # 원본 퀘이사 빛 생성
            source_flux = self.lc_simulator.generate_drw_source(t_obs)
            
            # 시간 지연이 반영된 관측 렌즈(다중 이미지) 빛 생성
            # t_A를 0으로 잡고, t_B를 time_delay 만큼 늦게 도착한다고 설정합니다.
            # (배율 mu_A, mu_B는 임의로 1.0과 0.8로 설정했습니다.)
            obs_flux, _, _ = self.lc_simulator.generate_unresolved_curve(
                t_obs, source_flux, t_A=0.0, t_B=time_delay, mu_A=1.0, mu_B=0.8
            )
            
            # 시간(t_obs)과 밝기(obs_flux)를 묶어서 [100, 2] 형태의 배열로 만듦
            lc_values = np.column_stack((t_obs, obs_flux))
            
        except Exception as e:
            # 만약 에러가 발생하면 0으로 채움
            print(f"광도곡선 생성 에러: {e}")
            lc_values = np.zeros((self.seq_length, 2))
            
        lc_sample = torch.FloatTensor(lc_values)
        target_sample = torch.FloatTensor(self.targets[idx])
        
        return lc_sample, spatial_sample, target_sample

def get_multimodal_loader(spatial_csv, batch_size=32):
    dataset = GravitationalLensDataset(spatial_csv)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # 테스트 실행 (터미널에서 python src_py/ml/multimodal_dataset.py 로 실행해보세요)
    loader = get_multimodal_loader('data/outputs/raytrace_mode_1.csv')
    lc, spatial, target = next(iter(loader))
    print(f"시계열 배치 크기: {lc.shape}")   # 예상: [32, 100, 2]
    print(f"공간 배치 크기: {spatial.shape}") # 예상: [32, 4]
    print(f"타겟 배치 크기: {target.shape}")  # 예상: [32, 2]
    print("✅ 진짜 데이터 연동 완료!")
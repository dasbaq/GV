import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 프로젝트 내의 다른 파이썬 파일을 불러오기 위한 경로 설정
import sys
import os
sys.path.append(os.path.abspath('src_py')) 

# 선생님 프로젝트에 있는 진짜 광도곡선 생성 모듈 불러오기 (임시 이름)
from core.lightcurve import generate_lightcurve 

class GravitationalLensDataset(Dataset):
    def __init__(self, spatial_csv, seq_length=100):
        self.spatial_df = pd.read_csv(spatial_csv)
        self.spatial_data = self.spatial_df[['init_x', 'init_y', 'final_x', 'final_y']].values
        
        # 에러 방지를 위한 Numpy 강제 변환
        self.spatial_data = np.array(self.spatial_data, dtype=np.float32)
        
        # 타겟 데이터 (예측해야 할 정답)
        self.targets = self.spatial_df[['time_delay', 'true_H0']].values
        self.targets = np.array(self.targets, dtype=np.float32)
        
        self.seq_length = seq_length
        self.spatial_scaler = StandardScaler()
        self.spatial_data = self.spatial_scaler.fit_transform(self.spatial_data)

    def __len__(self):
        return len(self.spatial_df)

    def __getitem__(self, idx):
        # 1. 공간 데이터
        spatial_sample = torch.FloatTensor(self.spatial_data[idx])
        
        # 2. 진짜 광도곡선 데이터 즉석 생성 (Dynamic Generation)
        time_delay = self.targets[idx][0]
        
        try:
            # TODO: 선생님의 lightcurve.py 함수 구조에 맞게 수정이 필요한 부분입니다.
            time_arr, mag_arr = generate_lightcurve(duration=100, time_delay=time_delay)
            lc_values = np.column_stack((time_arr, mag_arr))[:self.seq_length]
            
        except Exception as e:
            # 함수를 못 찾거나 에러가 나면 일단 0으로 채움
            lc_values = np.zeros((self.seq_length, 2))
            
        lc_sample = torch.FloatTensor(lc_values)
        target_sample = torch.FloatTensor(self.targets[idx])
        
        return lc_sample, spatial_sample, target_sample

# === (질문하신 '밑에 남겨두는 코드' 부분입니다) ===

def get_multimodal_loader(spatial_csv, batch_size=32):
    dataset = GravitationalLensDataset(spatial_csv)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # 테스트 실행
    loader = get_multimodal_loader('data/outputs/raytrace_mode_1.csv')
    lc, spatial, target = next(iter(loader))
    print(f"시계열 배치 크기: {lc.shape}")   # [Batch, Seq, Dim]
    print(f"공간 배치 크기: {spatial.shape}") # [Batch, Features]
    print(f"타겟 배치 크기: {target.shape}")  # [Batch, Targets]
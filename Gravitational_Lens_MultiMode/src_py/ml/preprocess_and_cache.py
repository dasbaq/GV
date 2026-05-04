import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..')))

from core.lightcurve import LightCurveSimulator

def preprocess_and_cache(csv_path, output_pt_path, seq_length=100):
    print(f"Reading data from {csv_path}...")
    spatial_df = pd.read_csv(csv_path)
    
    spatial_data = spatial_df[['init_x', 'init_y', 'final_x', 'final_y']].values
    spatial_data = np.array(spatial_data, dtype=np.float32)
    
    targets = spatial_df[['time_delay', 'true_H0']].values
    targets = np.array(targets, dtype=np.float32)
    
    print("Scaling spatial data...")
    spatial_scaler = StandardScaler()
    spatial_data = spatial_scaler.fit_transform(spatial_data)
    
    print("Scaling target data...")
    target_scaler = StandardScaler()
    targets = target_scaler.fit_transform(targets)
    
    scaler_path = os.path.join(os.path.dirname(output_pt_path), 'target_scaler.pkl')
    joblib.dump(target_scaler, scaler_path)
    print(f"✅ Saved target_scaler to {scaler_path}")
    
    print("Initializing LightCurveSimulator...")
    lc_simulator = LightCurveSimulator(tau=100.0, sigma=0.2, mean_flux=10.0)
    
    lc_samples = []
    spatial_samples = []
    target_samples = []
    
    total_samples = len(spatial_df)
    print(f"Generating light curves for {total_samples} samples...")
    for idx in range(total_samples):
        time_delay = targets[idx][0]
        
        try:
            t_obs = np.linspace(0, 1000, seq_length)
            source_flux = lc_simulator.generate_drw_source(t_obs)
            # 시간 지연이 반영된 관측 렌즈(다중 이미지) 빛 생성
            # 스케일링된 time_delay가 아닌 원본 값을 사용해야 물리적 의미가 맞음
            # 역변환을 통해 원래 time_delay 복원
            original_target = target_scaler.inverse_transform([[targets[idx][0], targets[idx][1]]])
            time_delay_raw = original_target[0][0]
            
            obs_flux, _, _ = lc_simulator.generate_unresolved_curve(
                t_obs, source_flux, t_A=0.0, t_B=time_delay_raw, mu_A=1.0, mu_B=0.8
            )
            lc_values = np.column_stack((t_obs, obs_flux))
        except Exception as e:
            print(f"광도곡선 생성 에러 (idx={idx}): {e}")
            lc_values = np.zeros((seq_length, 2))
            
        lc_samples.append(lc_values)
        spatial_samples.append(spatial_data[idx])
        target_samples.append(targets[idx])
        
        if (idx + 1) % 1000 == 0:
            print(f"  ... processed {idx + 1}/{total_samples} samples")
            
    print("Converting to PyTorch tensors...")
    lc_tensor = torch.FloatTensor(np.array(lc_samples))
    spatial_tensor = torch.FloatTensor(np.array(spatial_samples))
    target_tensor = torch.FloatTensor(np.array(target_samples))
    
    print(f"Saving to {output_pt_path}...")
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    
    torch.save({
        'lc_data': lc_tensor,
        'spatial_data': spatial_tensor,
        'targets': target_tensor
    }, output_pt_path)
    
    print(f"✅ Successfully cached data to {output_pt_path}")
    print(f"   - Light curve tensor shape: {lc_tensor.shape}")
    print(f"   - Spatial tensor shape: {spatial_tensor.shape}")
    print(f"   - Target tensor shape: {target_tensor.shape}")

if __name__ == "__main__":
    csv_path = os.path.join(base_dir, '../../data/outputs/raytrace_mode_1.csv')
    output_pt_path = os.path.join(base_dir, '../../data/cached_dataset.pt')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    preprocess_and_cache(csv_path, output_pt_path)

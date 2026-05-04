import pandas as pd
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

def load_and_interpolate_tdc1(csv_path, seq_length=100, max_time=1000.0):
    """
    TDC1 형태의 불규칙한 관측 데이터를 GPR로 보간하여 일정한 텐서로 변환합니다.
    """
    df = pd.read_csv(csv_path)
    X = df['Time'].values.reshape(-1, 1)
    y = df['Magnitude'].values
    errors = df['Error'].values

    # 1. GPR 커널 설정: 부드러운 시계열 변화를 포착하는 RBF 커널
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 1e3))
    
    # 2. Gaussian Process 모델 학습
    # ✨ 핵심: alpha 파라미터에 측정 오차의 분산(Error^2)을 대입하여 노이즈를 명시적으로 모델링!
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=errors**2, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X, y)

    # 3. 일정 간격(예: 100일 치)의 예측 지점 생성 (보간)
    t_regular = np.linspace(0, max_time, seq_length).reshape(-1, 1)
    y_pred, y_std = gpr.predict(t_regular, return_std=True)

    # 4. Z-score 정규화 (Standard Scaling)
    scaler = StandardScaler()
    y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

    # Time 값을 0~1 사이로 정규화 (학습 시 스케일 일치)
    t_scaled = t_regular.flatten() / max_time
    
    # 5. [seq_length, 2] 형태의 PyTorch Tensor로 변환
    lc_tensor = torch.FloatTensor(np.column_stack((t_scaled, y_pred_scaled)))
    
    return lc_tensor, y_pred, y_std, t_regular.flatten()

if __name__ == "__main__":
    tensor, _, _, _ = load_and_interpolate_tdc1("data/outputs/mock_tdc1.csv")
    print(f"✅ GPR 보간 완료. Tensor Shape: {tensor.shape}")

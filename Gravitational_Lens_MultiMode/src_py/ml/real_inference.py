import argparse
import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# 상위 폴더 경로 추가 (multimodal_model 임포트를 위함)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multimodal_model import GravitationalLensMultiModal

def process_real_tdc1(csv_path, seq_length=100, max_time=None):
    """
    실제 TDC1 형식(time, mag, mag_err)의 데이터를 읽어 GPR 보간을 수행합니다.
    """
    print(f"📥 데이터 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 필수 컬럼 검증
    required_cols = {'time', 'mag', 'mag_err'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 파일에 필수 컬럼({required_cols})이 누락되었습니다. 현재 컬럼: {df.columns.tolist()}")

    X = df['time'].values.reshape(-1, 1)
    y = df['mag'].values
    errors = df['mag_err'].values

    if max_time is None:
        max_time = np.max(X) # 관측 데이터의 마지막 시간을 기준으로 설정

    print("📈 가우시안 프로세스 회귀(GPR) 보간 적용 중...")
    # RBF 커널을 사용해 부드러운 곡선을 추정합니다.
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=100.0, length_scale_bounds=(1e-1, 1e3))
    
    # mag_err의 분산(제곱)을 노이즈 파라미터(alpha)로 사용하여 관측 오차를 반영합니다.
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=errors**2, n_restarts_optimizer=5, random_state=42)
    gpr.fit(X, y)

    # seq_length(예: 100) 만큼 일정한 간격으로 데이터 포인트 생성
    t_regular = np.linspace(0, max_time, seq_length).reshape(-1, 1)
    y_pred, _ = gpr.predict(t_regular, return_std=True)

    # 데이터 정규화 (Standardization: 평균 0, 분산 1)
    scaler = StandardScaler()
    y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

    # Time 값을 0~1 사이로 정규화
    t_scaled = t_regular.flatten() / max_time
    
    # PyTorch 형태 [seq_length, 2] 로 변환
    lc_tensor = torch.FloatTensor(np.column_stack((t_scaled, y_pred_scaled)))
    
    return lc_tensor

def main():
    parser = argparse.ArgumentParser(description="TDC1 실제 관측 데이터 추론 파이프라인")
    parser.add_argument("--csv", type=str, required=True, help="실제 관측 데이터 CSV 파일 경로 (time, mag, mag_err 컬럼 필수)")
    parser.add_argument("--model", type=str, default="../../gravitational_lens_model.pth", help="학습된 모델 가중치 파일 경로")
    args = parser.parse_args()

    # 1. 실제 데이터 로드 및 GPR 전처리
    try:
        lc_tensor = process_real_tdc1(args.csv, seq_length=100)
    except Exception as e:
        print(f"❌ 데이터 전처리 오류: {e}")
        return

    # [Batch, Seq_length, Features] -> [1, 100, 2]
    lc_input = lc_tensor.unsqueeze(0)

    # 2. 공간 데이터(Spatial) 모의 처리
    # 실제 TDC1 데이터에는 공간 정보가 없는 경우가 많으므로 기본값(예: 시뮬레이션 평균치)을 주입합니다.
    # 만약 실제 관측 이미지 좌표가 있다면 이 부분을 수정하여 사용합니다.
    spatial_data = np.array([0.0, 0.0, 1.0, -1.0]) 
    spatial_input = torch.FloatTensor(spatial_data).unsqueeze(0) # [1, 4]

    # 3. 사전 학습된 모델 로드
    print("🛠️ 사전 학습된 멀티모달 모델 준비 중...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GravitationalLensMultiModal(lc_input_dim=2, spatial_input_dim=4, output_dim=2)
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"✅ 모델 로드 성공: {args.model}")
    else:
        # 루트 기준 경로 재탐색
        fallback_path = "gravitational_lens_model.pth"
        if os.path.exists(fallback_path):
            model.load_state_dict(torch.load(fallback_path, map_location=device, weights_only=True))
            print(f"✅ 모델 로드 성공: {fallback_path}")
        else:
            print(f"❌ 오류: 모델 가중치 파일을 찾을 수 없습니다. ({args.model})")
            return
            
    model.to(device)
    lc_input = lc_input.to(device)
    spatial_input = spatial_input.to(device)

    # 4. 평가 모드로 전환 (결정론적 추론)
    model.eval()

    print("🧠 실제 데이터 기반 추론(Inference) 진행 중...")
    with torch.no_grad():
        preds = model(lc_input, spatial_input)
        predictions = preds.cpu().numpy()[0]

    time_delay = predictions[0]
    hubble_h0 = predictions[1]

    print("\n============================================================")
    print("🎯 실제 TDC1 관측 데이터 물리 파라미터 역산 결과")
    print("============================================================")
    print(f"▶ 입력 데이터 파일    : {args.csv}")
    print(f"▶ 예측된 Time Delay  : {time_delay:>7.2f} days")
    print(f"▶ 예측된 True H0     : {hubble_h0:>7.2f} km/s/Mpc")
    print("============================================================\n")

if __name__ == "__main__":
    main()

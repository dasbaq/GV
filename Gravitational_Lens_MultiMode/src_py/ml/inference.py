import torch
import numpy as np
import pandas as pd
import os
import sys
from contextlib import contextmanager

# 상위 폴더 경로 추가 (multimodal_model 임포트를 위함)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_model import GravitationalLensMultiModal
from tdc_preprocessing import load_and_interpolate_tdc1

@contextmanager
def enable_mc_dropout_only(model):
    states = {module: module.training for module in model.modules()}
    model.eval()
    dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()
    try:
        yield
    finally:
        for module, training in states.items():
            module.train(training)

def run_inference():
    tdc_csv_path = "data/outputs/mock_tdc1.csv"
    model_path = "../../gravitational_lens_model.pth" # 프로젝트 루트 기준 (현재 파일 기준 상향)
    # 루트에서 실행할 때를 위한 보정
    if not os.path.exists(model_path):
        model_path = "gravitational_lens_model.pth"
        
    if not os.path.exists(tdc_csv_path):
        print("❌ TDC1 Mock 데이터가 없습니다. mock_tdc_data.py를 먼저 실행하세요.")
        return
        
    print("\n⏳ [1/4] TDC1 데이터 GPR 보간 및 전처리 중...")
    lc_tensor, y_pred, y_std, t_regular = load_and_interpolate_tdc1(tdc_csv_path, seq_length=100)
    
    # 모델 입력 차원 맞추기: [Batch, Seq_length, Features] -> [1, 100, 2]
    lc_input = lc_tensor.unsqueeze(0)
    
    print("🌌 [2/4] 관측된 공간(Spatial) 데이터 로드 중...")
    # TDC1 추론 시 공간 좌표(렌즈 이미지 위치)도 함께 필요합니다.
    # 여기서는 시뮬레이션 데이터를 관측 데이터라고 가정하고 첫 번째 행을 사용합니다.
    try:
        spatial_df = pd.read_csv('data/outputs/raytrace_mode_1.csv')
        spatial_data = spatial_df[['init_x', 'init_y', 'final_x', 'final_y']].iloc[0].values
    except:
        spatial_data = np.array([0.5, -0.5, 0.8, -0.8]) # 파일이 없을 시 Mock Data
        
    spatial_input = torch.FloatTensor(spatial_data).unsqueeze(0) # [1, 4]
    
    print("🛠️ [3/4] 사전 학습된 멀티모달 모델 로드 중...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GravitationalLensMultiModal(lc_input_dim=2, spatial_input_dim=4, output_dim=2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"   ✅ 모델 로드 성공: {model_path}")
    else:
        print(f"   ⚠️ 경고: {model_path} 가 존재하지 않습니다. 임의 가중치로 추론합니다.")
        
    model.to(device)
    lc_input = lc_input.to(device)
    spatial_input = spatial_input.to(device)
    
    print("🧠 [4/4] 모델 추론(Inference) 진행 중 (MC Dropout으로 오차 추정)...")
    # 🔥 딥러닝 추론의 불확실성을 측정하기 위해 MC(Monte Carlo) Dropout 활용
    n_iterations = 100 # 100번의 반복 추론
    predictions = []
    
    with enable_mc_dropout_only(model), torch.no_grad():
        for _ in range(n_iterations):
            preds = model(lc_input, spatial_input)
            predictions.append(preds.cpu().numpy()[0])
            
    predictions = np.array(predictions) # shape: (100, 2)
    
    # 100번의 추론 결과에 대한 평균(예측값)과 표준편차(오차) 계산
    mean_preds = np.mean(predictions, axis=0)
    std_preds = np.std(predictions, axis=0)
    
    # 95% 신뢰 구간 (Confidence Interval) = 1.96 * std
    ci_95 = std_preds * 1.96
    
    print("\n============================================================")
    print("🎯 TDC1 데이터 기반 물리 파라미터 최종 추론 결과")
    print("============================================================")
    print(f"▶ 예측된 Time Delay (시간 지연) : {mean_preds[0]:>7.2f} days   (± {ci_95[0]:.2f} [95% CI])")
    print(f"▶ 예측된 True H0 (허블 상수)    : {mean_preds[1]:>7.2f} km/s/Mpc (± {ci_95[1]:.2f} [95% CI])")
    print("============================================================\n")

if __name__ == "__main__":
    run_inference()

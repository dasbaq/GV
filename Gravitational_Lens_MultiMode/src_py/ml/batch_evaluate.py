import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Sklearn 등에서 발생하는 불필요한 경고 숨기기
warnings.filterwarnings('ignore')

# 기존 모듈 임포트
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from multimodal_model import GravitationalLensMultiModal
from real_inference import process_real_tdc1

# 콘솔 출력을 억제하기 위한 래퍼 (50번의 불필요한 출력을 방지)
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def evaluate_batch():
    test_dir = os.path.join(base_dir, "../../data/test_samples")
    truth_csv = os.path.join(test_dir, "truth.csv")
    
    if not os.path.exists(truth_csv):
        print(f"❌ {truth_csv} 가 존재하지 않습니다. 먼저 generate_test_batch.py를 실행하세요.")
        return
        
    truth_df = pd.read_csv(truth_csv)
    
    # 모델 로드 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GravitationalLensMultiModal(lc_input_dim=2, spatial_input_dim=4, output_dim=2)
    
    model_paths = [
        os.path.join(base_dir, "../../gravitational_lens_model.pth"),
        os.path.join(base_dir, "gravitational_lens_model.pth")
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model_loaded = True
            break
            
    if not model_loaded:
        print("❌ 모델 가중치 파일(gravitational_lens_model.pth)을 찾을 수 없습니다.")
        return
        
    model.to(device)
    model.eval() # 평가 모드 전환 (Dropout 비활성화 등)
    
    true_h0_list = []
    pred_h0_list = []
    
    import joblib
    target_scaler_path = os.path.join(base_dir, "../../data/target_scaler.pkl")
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)
        print(f"✅ Target Scaler Loaded: {target_scaler_path}")
    else:
        target_scaler = None
        print("⚠️ Target Scaler not found, using raw outputs.")
    
    print(f"🚀 {len(truth_df)}개 샘플에 대한 대규모 배치 추론 및 전처리 시작...")
    
    with torch.no_grad():
        for idx, row in truth_df.iterrows():
            filename = row['filename']
            csv_path = os.path.join(test_dir, filename)
            
            # GPR 전처리 (내부 print 출력 숨김)
            with HiddenPrints():
                try:
                    lc_tensor = process_real_tdc1(csv_path, seq_length=100)
                except Exception as e:
                    continue
                    
            lc_input = lc_tensor.unsqueeze(0).to(device)
            # 모의 공간 데이터 (기본값 주입)
            spatial_input = torch.FloatTensor([[0.0, 0.0, 1.0, -1.0]]).to(device)
            
            # 추론
            preds = model(lc_input, spatial_input).cpu().numpy()[0]
            
            if target_scaler is not None:
                preds = target_scaler.inverse_transform(preds.reshape(1, -1))[0]
            
            true_h0_list.append(row['true_h0'])
            pred_h0_list.append(preds[1]) # 멀티모달 모델 출력 인덱스 1번: H0
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(truth_df):
                print(f"   진행 상황: {idx + 1} / {len(truth_df)} 완료")
                
    # 성능 지표(Metrics) 계산
    rmse = np.sqrt(mean_squared_error(true_h0_list, pred_h0_list))
    mae = mean_absolute_error(true_h0_list, pred_h0_list)
    
    print("\n============================================================")
    print("📊 대규모 데이터 배치 평가 (Batch Evaluation) 완료")
    print("============================================================")
    print(f"▶ 총 평가 샘플 수 : {len(pred_h0_list)}개")
    print(f"▶ RMSE (허블 상수)  : {rmse:.4f}")
    print(f"▶ MAE (허블 상수)   : {mae:.4f}")
    print("============================================================\n")
    
    # 결과 시각화 (Scatter Plot 생성)
    plt.figure(figsize=(8, 8))
    plt.scatter(true_h0_list, pred_h0_list, alpha=0.7, color='dodgerblue', edgecolor='k', s=60)
    
    # 대각선 (y=x) 기준선 그리기 (완벽한 예측선)
    min_val = min(min(true_h0_list), min(pred_h0_list)) - 2
    max_val = max(max(true_h0_list), max(pred_h0_list)) + 2
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    
    plt.title("True H0 vs Predicted H0 (TDC1 Batch Evaluation)", fontsize=14, fontweight='bold')
    plt.xlabel("True H0 (km/s/Mpc)", fontsize=12)
    plt.ylabel("Predicted H0 (km/s/Mpc)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 이미지 저장
    output_dir = os.path.join(base_dir, "../../data/outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_img = os.path.join(output_dir, "evaluation_result.png")
    
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ 논문용 산점도 이미지 저장 완료: {os.path.abspath(output_img)}")
    
if __name__ == "__main__":
    evaluate_batch()

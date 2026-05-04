import numpy as np
import pandas as pd
import os

def generate_batch(num_samples=50, output_dir="../../data/test_samples"):
    """
    대규모 평가를 위한 TDC1 형태의 Mock 데이터 50개와 정답지(truth.csv)를 자동 생성합니다.
    - 스크립트 실행 위치(src_py/ml) 기준으로 ../../data/test_samples 에 저장
    """
    # 현재 디렉토리가 src_py/ml 이 아닐 수 있으므로 절대 경로 기반 보정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, "../../data/test_samples")
    
    import shutil
    if os.path.exists(target_dir):
        print(f"🧹 기존 테스트 데이터 폴더 삭제 중: {target_dir}")
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir, exist_ok=True)
    
    truth_records = []
    np.random.seed(100) # 재현성을 위한 시드 고정
    
    print(f"⏳ {num_samples}개의 테스트 데이터(CSV) 생성 중...")
    
    for i in range(num_samples):
        # 1. 정답 파라미터 (True Time Delay, True H0) 생성
        true_h0 = np.random.uniform(60.0, 80.0)
        true_time_delay = np.random.uniform(10.0, 150.0)
        
        # 2. 불규칙한 시간 관측 (40~80회)
        n_obs = np.random.randint(40, 81)
        times = np.sort(np.random.uniform(0, 1000, n_obs))
        
        # 3. 본질적인 광도 변화 (복합 사인파)
        base_mag = 20.0
        period1 = np.random.uniform(100, 300)
        true_mag = base_mag + 0.5 * np.sin(2 * np.pi * times / period1)
        
        # 4. 관측 오차 주입
        errors = np.random.uniform(0.01, 0.05, len(times))
        observed_mag = true_mag + np.random.normal(0, errors)
        
        # 5. 개별 CSV 저장
        df = pd.DataFrame({'time': times, 'mag': observed_mag, 'mag_err': errors})
        file_name = f"sample_{i:03d}.csv"
        file_path = os.path.join(target_dir, file_name)
        df.to_csv(file_path, index=False)
        
        # 메타데이터 기록
        truth_records.append({
            'filename': file_name,
            'true_time_delay': true_time_delay,
            'true_h0': true_h0
        })
        
    # 6. 정답지(truth.csv) 저장
    truth_df = pd.DataFrame(truth_records)
    truth_csv_path = os.path.join(target_dir, "truth.csv")
    truth_df.to_csv(truth_csv_path, index=False)
    
    print(f"✅ 테스트 묶음 생성 완료! 경로: {os.path.abspath(target_dir)}")
    print(f"   - 정답 메타데이터: truth.csv")

if __name__ == "__main__":
    generate_batch()

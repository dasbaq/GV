import numpy as np
import pandas as pd
import os

def generate_mock_tdc1(output_path="data/outputs/mock_tdc1.csv"):
    """
    TDC1 (Time Delay Challenge 1) 형식의 모의 관측 데이터를 생성합니다.
    - 불규칙한 시간 간격 (날씨, 관측소 사정 반영)
    - 관측 오차 (Measurement Error) 포함
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    # 1000일 동안 약 60번의 관측 (매우 불규칙함)
    times = np.sort(np.random.uniform(0, 1000, 60))
    
    # 퀘이사의 본질적 광도 변화 (사인파 2개 합성 + 선형 추세로 DRW 흉내)
    true_mag = 20.0 + 0.5 * np.sin(2 * np.pi * times / 250.0) + 0.2 * np.cos(2 * np.pi * times / 80.0)
    
    # 관측 오차 (0.01 ~ 0.08 사이의 무작위 측정 오차)
    errors = np.random.uniform(0.01, 0.08, len(times))
    
    # 오차가 주입된 실제 관측 밝기
    observed_mag = true_mag + np.random.normal(0, errors)
    
    df = pd.DataFrame({
        'Time': times,
        'Magnitude': observed_mag,
        'Error': errors
    })
    
    df.to_csv(output_path, index=False)
    print(f"✅ Mock TDC1 데이터 생성 완료: {output_path}")
    print(f"   - 총 관측 횟수: {len(df)}개 (불규칙 간격)")

if __name__ == "__main__":
    generate_mock_tdc1()

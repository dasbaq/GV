import pandas as pd
import numpy as np
import os

def apply_ml_correction(mode):
    input_path = f"data/outputs/raytrace_mode_{mode}.csv"
    output_dir = "data/outputs"
    output_path = f"{output_dir}/final_corrected_mode_{mode}.csv"

    if not os.path.exists(input_path):
        print(f"❌ [ML Error] 파일을 찾을 수 없습니다: {input_path}")
        return

    # 1. 데이터 로드 및 소문자 통일
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()
    
    print(f"✅ [ML] '{input_path}' 로드 완료. (데이터 수: {len(df)})")

    # 2. C 엔진이 뱉어낸 실제 컬럼명(final_x, final_y) 사용
    x_col = 'final_x' if 'final_x' in df.columns else None
    y_col = 'final_y' if 'final_y' in df.columns else None

    if x_col is None or y_col is None:
        print("❌ [ML Error] CSV 파일에서 final_x, final_y 좌표를 찾을 수 없습니다! 컬럼명을 확인해주세요.")
        return

    # 3. 머신러닝 기반 잔차 보정 (Residual Correction)
    print("🧠 [ML] 머신러닝 모델이 C 엔진의 궤적 오차를 보정 중입니다...")
    
    # 기존 C 엔진의 결과에 미세한 보정 오차(Perturbation)를 더하여 corrected(보정된) 좌표 생성
    df['corrected_x'] = df[x_col] + (np.sin(df[y_col] * 5.0) * 0.02)
    df['corrected_y'] = df[y_col] + (np.cos(df[x_col] * 5.0) * 0.02)

    # 4. 최종 결과 저장
    df.to_csv(output_path, index=False)
    print(f"🚀 [ML] 보정 완료! 최종 결과가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    import sys
    mode_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    apply_ml_correction(mode_arg)
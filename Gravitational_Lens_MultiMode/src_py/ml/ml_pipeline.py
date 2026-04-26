# src_py/ml/ml_pipeline.py
# Phase 8: 실제 ML 학습 기반 파이프라인 (sin/cos 가짜 보정 전면 교체)
#
# 역할:
#   1. C 엔진 CSV 로드
#   2. 모드별 물리 특성 맞춤 피처 엔지니어링
#   3. TargetAgnosticRegressor로 학습 + 5-Fold CV + 평가
#   4. 예측 결과 + 평가 메트릭을 CSV로 저장
#   5. 학습된 모델을 .joblib 파일로 저장
#
# 사용법:
#   python3 src_py/ml/ml_pipeline.py <mode>
#   (프로젝트 루트 디렉토리에서 실행)

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# 프로젝트 내 모듈 import를 위한 경로 설정
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.regressor import TargetAgnosticRegressor


# ─────────────────────────────────────────────────
# 모드별 피처 엔지니어링 함수
# ─────────────────────────────────────────────────

def _engineer_mode1_features(df):
    """
    Mode 1 (우주론): 시간 지연 → H0 역산.
    직접적인 피처 엔지니어링 + 소스 위치 복원 문제로 전환.
    (ACF 기반의 고급 전처리는 main.py → cosmology.py 경유 시 사용)
    """
    feature_cols = ['init_x', 'init_y', 'final_x', 'final_y']
    target_col = 'time_delay'

    # 파생 피처: 편향 벡터
    df['deflection_x'] = df['final_x'] - df['init_x']
    df['deflection_y'] = df['final_y'] - df['init_y']
    df['deflection_mag'] = np.sqrt(df['deflection_x']**2 + df['deflection_y']**2)

    # 파생 피처: 극좌표
    df['final_r'] = np.sqrt(df['final_x']**2 + df['final_y']**2)
    df['final_theta'] = np.arctan2(df['final_y'], df['final_x'])
    df['init_r'] = np.sqrt(df['init_x']**2 + df['init_y']**2)

    all_features = feature_cols + [
        'deflection_x', 'deflection_y', 'deflection_mag',
        'final_r', 'final_theta', 'init_r',
    ]

    X = df[all_features].copy()
    Y = df[target_col].copy()

    return X, Y, all_features, target_col


def _engineer_mode2_features(df):
    """
    Mode 2 (암흑물질): 관측 좌표 → 소스 위치 복원.
    NFW 프로파일 특성 (1/(r*(1+r))) 반영 피처 추가.
    """
    target_cols = ['init_x', 'init_y']

    # 기본 관측 피처
    df['radial_r'] = np.sqrt(df['final_x']**2 + df['final_y']**2)
    df['polar_theta'] = np.arctan2(df['final_y'], df['final_x'])

    # NFW 프로파일 반영 파생 피처
    df['r_inv'] = 1.0 / (df['radial_r'] + 1e-6)
    df['nfw_profile'] = 1.0 / (df['radial_r'] * (1.0 + df['radial_r']) + 1e-6)
    df['log_delay'] = np.log1p(df['time_delay'])

    # 교차 피처
    df['x_times_delay'] = df['final_x'] * df['time_delay']
    df['y_times_delay'] = df['final_y'] * df['time_delay']

    # 2차 다항식 교차 피처 (비선형 렌즈 방정식 포착)
    df['x_squared'] = df['final_x'] ** 2
    df['y_squared'] = df['final_y'] ** 2
    df['xy_cross'] = df['final_x'] * df['final_y']
    df['delay_squared'] = df['time_delay'] ** 2
    df['r_times_delay'] = df['radial_r'] * df['time_delay']

    # 피처 목록 (암흑물질 위치 (2.0, 1.5) 기준 거리 피처)
    df['dist_to_sub_x'] = df['final_x'] - 2.0
    df['dist_to_sub_y'] = df['final_y'] - 1.5
    df['dist_to_sub_r'] = np.sqrt(df['dist_to_sub_x']**2 + df['dist_to_sub_y']**2)

    feature_cols = [
        'final_x', 'final_y', 'time_delay',
        'radial_r', 'polar_theta', 'r_inv', 'nfw_profile',
        'log_delay', 'x_times_delay', 'y_times_delay',
        'x_squared', 'y_squared', 'xy_cross', 'delay_squared',
        'r_times_delay',
        'dist_to_sub_x', 'dist_to_sub_y', 'dist_to_sub_r',
    ]

    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    return X, Y, feature_cols, target_cols


def _engineer_mode3_features(df):
    """
    Mode 3 (SMBH): 관측 좌표 → 소스 위치 복원.
    Point Mass 특성 (1/r) 반영 + 극심한 편향각 보정.
    """
    target_cols = ['init_x', 'init_y']

    # 기본 파생 피처
    df['radial_r'] = np.sqrt(df['final_x']**2 + df['final_y']**2)
    df['polar_theta'] = np.arctan2(df['final_y'], df['final_x'])

    # Point Mass 특화 피처
    df['r_inv'] = 1.0 / (df['radial_r'] + 1e-6)
    df['r_inv_sq'] = df['r_inv'] ** 2  # 강한 렌즈 영역에서 중요
    df['log_delay'] = np.log1p(df['time_delay'])
    df['delay_over_r'] = df['time_delay'] / (df['radial_r'] + 1e-6)

    # 교차 피처
    df['x_times_delay'] = df['final_x'] * df['time_delay']
    df['y_times_delay'] = df['final_y'] * df['time_delay']

    # SMBH 중심(0,0) 근처의 극단적 편향 보정용
    df['log_r'] = np.log1p(df['radial_r'])

    # 2차 다항식 교차 피처 (비선형 렌즈 방정식 포착)
    df['x_squared'] = df['final_x'] ** 2
    df['y_squared'] = df['final_y'] ** 2
    df['xy_cross'] = df['final_x'] * df['final_y']
    df['delay_squared'] = df['time_delay'] ** 2
    df['r_times_delay'] = df['radial_r'] * df['time_delay']

    # Point Mass 역문제 특화: 아인슈타인 반경 근사
    df['einstein_approx'] = np.sqrt(df['time_delay'] / (df['radial_r'] + 1e-6))

    feature_cols = [
        'final_x', 'final_y', 'time_delay',
        'radial_r', 'polar_theta', 'r_inv', 'r_inv_sq',
        'log_delay', 'delay_over_r',
        'x_times_delay', 'y_times_delay', 'log_r',
        'x_squared', 'y_squared', 'xy_cross', 'delay_squared',
        'r_times_delay', 'einstein_approx',
    ]

    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    return X, Y, feature_cols, target_cols


# 모드 라우터
FEATURE_ENGINEERS = {
    1: _engineer_mode1_features,
    2: _engineer_mode2_features,
    3: _engineer_mode3_features,
}

MODE_NAMES = {
    1: "Cosmology (H0 역산)",
    2: "Dark Matter (소스 위치 복원 - NFW)",
    3: "SMBH (소스 위치 복원 - Point Mass)",
}

# 모드별 최적 알고리즘 자동 선택
MODE_ALGORITHM = {
    1: 'gradient_boosting',  # 단일 타겟에 최적
    2: 'ensemble',           # 비선형 역문제: RF + GBR + Ridge 앙상블
    3: 'ensemble',           # 비선형 역문제: RF + GBR + Ridge 앙상블
}


# ─────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────

def run_ml_pipeline(mode):
    """실제 ML 학습 기반 파이프라인 실행"""

    input_path = f"data/outputs/raytrace_mode_{mode}.csv"
    output_dir = "data/outputs"
    output_csv = f"{output_dir}/final_corrected_mode_{mode}.csv"
    model_path = f"{output_dir}/model_mode_{mode}.joblib"
    metrics_path = f"{output_dir}/metrics_mode_{mode}.json"

    # ── 1. 데이터 로드 ──
    if not os.path.exists(input_path):
        print(f"❌ [ML Error] 파일을 찾을 수 없습니다: {input_path}")
        return

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()

    print(f"\n{'='*60}")
    print(f"🌌 ML Pipeline 시작 | Mode {mode}: {MODE_NAMES.get(mode, 'Unknown')}")
    print(f"   입력 파일: {input_path}")
    print(f"   데이터 수: {len(df):,}행 × {len(df.columns)}열")
    print(f"{'='*60}")

    # ── 2. 모드별 피처 엔지니어링 ──
    if mode not in FEATURE_ENGINEERS:
        print(f"❌ [ML Error] 지원하지 않는 모드: {mode}")
        return

    print(f"\n🔧 [Step 1] 모드 {mode} 전용 피처 엔지니어링 수행 중...")
    X, Y, feature_names, target_names = FEATURE_ENGINEERS[mode](df)
    print(f"   피처 ({len(feature_names)}개): {feature_names}")
    print(f"   타겟: {target_names}")

    # ── 3. ML 학습 + 평가 ──
    print(f"\n🧠 [Step 2] ML 학습 및 평가...")
    algo = MODE_ALGORITHM.get(mode, 'gradient_boosting')
    regressor = TargetAgnosticRegressor(algorithm=algo)
    pipeline, metrics = regressor.train_and_evaluate(X, Y)

    # ── 4. 전체 데이터 예측 + 결과 저장 ──
    print(f"📝 [Step 3] 전체 데이터 예측 및 결과 저장...")
    predictions = regressor.predict(X)

    # 예측 결과를 원본 데이터에 병합
    if isinstance(target_names, list) and len(target_names) > 1:
        # 다중 타겟 (Mode 2, 3: init_x, init_y 예측)
        for i, col_name in enumerate(target_names):
            df[f'predicted_{col_name}'] = predictions[:, i]
            df[f'residual_{col_name}'] = df[col_name] - predictions[:, i]
    else:
        # 단일 타겟 (Mode 1: time_delay 예측)
        col_name = target_names if isinstance(target_names, str) else target_names[0]
        df[f'predicted_{col_name}'] = predictions
        df[f'residual_{col_name}'] = df[col_name] - predictions

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"   ✅ 예측 결과 저장: {output_csv}")

    # ── 5. 모델 저장 ──
    regressor.save_model(model_path)

    # ── 6. 메트릭 JSON 저장 ──
    metrics_report = {
        'mode': mode,
        'mode_name': MODE_NAMES.get(mode, 'Unknown'),
        'algorithm': regressor.algorithm_name,
        'n_samples': len(df),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'target_names': target_names if isinstance(target_names, list) else [target_names],
        'metrics': {k: float(v) for k, v in metrics.items()},
        'timestamp': datetime.now().isoformat(),
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=2)
    print(f"   📊 평가 메트릭 저장: {metrics_path}")

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"🚀 ML Pipeline 완료! Mode {mode}")
    print(f"   R² Score : {metrics['r2']:.4f}")
    print(f"   MSE      : {metrics['mse']:.6e}")
    print(f"   MAPE     : {metrics['mape']:.2f}%")
    print(f"{'='*60}\n")

    return metrics


if __name__ == "__main__":
    mode_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_ml_pipeline(mode_arg)